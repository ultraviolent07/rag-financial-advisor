"""
Regime Detection Module — Tier 3
Classifies market into trend / revert / volatile using HMM + KMeans on 90d price data.
Labels feed into LLM prompt as market context.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional
from datetime import datetime

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not available, using KMeans only")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============= Feature Engineering =============

def compute_regime_features(ticker: str, period: str = "90d") -> Optional[pd.DataFrame]:
    """
    Compute features used for regime classification:
    - Daily returns
    - 5-day rolling volatility
    - RSI (14-day)
    - Bollinger Band width (20-day)
    - Volume z-score
    - 10-day momentum
    """
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty or len(data) < 20:
            return None

        df = pd.DataFrame()
        close = data["Close"].squeeze()
        volume = data["Volume"].squeeze()

        # Returns
        df["returns"] = close.pct_change()

        # Rolling volatility (5-day)
        df["volatility"] = df["returns"].rolling(5).std()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Band width
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_width"] = (2 * std20) / sma20.replace(0, np.nan)

        # Volume z-score
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        df["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

        # Momentum (10-day)
        df["momentum"] = close.pct_change(10)

        df = df.dropna()
        return df

    except Exception as e:
        print(f"Feature computation error for {ticker}: {e}")
        return None


# ============= HMM Regime Classifier =============

class HMMRegimeClassifier:
    """
    Gaussian HMM with 3 hidden states → trend / revert / volatile
    Trained fresh on each ticker's 90d data (no pretrained weights needed).
    """

    N_STATES = 3
    FEATURES = ["returns", "volatility", "rsi", "bb_width", "volume_zscore", "momentum"]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_map = {}  # state_id -> regime label

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Fit HMM on features and return state sequence"""
        X = df[self.FEATURES].values
        X_scaled = self.scaler.fit_transform(X)

        model = GaussianHMM(
            n_components=self.N_STATES,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        model.fit(X_scaled)
        self.model = model
        states = model.predict(X_scaled)

        # Map states to regime labels based on volatility characteristic
        self._map_states_to_labels(df, states)
        return states

    def _map_states_to_labels(self, df: pd.DataFrame, states: np.ndarray):
        """
        Assign human-readable labels to HMM states:
        - Lowest volatility + positive returns → trend
        - Highest volatility → volatile
        - Middle → revert (mean-reverting)
        """
        state_stats = {}
        for s in range(self.N_STATES):
            mask = states == s
            if mask.sum() == 0:
                continue
            state_stats[s] = {
                "vol":     df["volatility"].values[mask].mean(),
                "returns": df["returns"].values[mask].mean(),
                "count":   mask.sum()
            }

        sorted_by_vol = sorted(state_stats.keys(), key=lambda s: state_stats[s]["vol"])

        if len(sorted_by_vol) == 3:
            low_vol, mid_vol, high_vol = sorted_by_vol
            # Low vol + positive return = trending
            if state_stats[low_vol]["returns"] > 0:
                self.label_map = {low_vol: "trend", mid_vol: "revert", high_vol: "volatile"}
            else:
                self.label_map = {low_vol: "revert", mid_vol: "trend", high_vol: "volatile"}
        elif len(sorted_by_vol) == 2:
            low_vol, high_vol = sorted_by_vol
            self.label_map = {low_vol: "trend", high_vol: "volatile"}
        else:
            for s in sorted_by_vol:
                self.label_map[s] = "revert"

    def get_current_regime(self, states: np.ndarray) -> str:
        """Return the most recent state label"""
        current_state = int(states[-1])
        return self.label_map.get(current_state, "unknown")

    def get_regime_probabilities(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get probability of each regime for the current observation"""
        if self.model is None:
            return {}
        try:
            X = df[self.FEATURES].values[-1:]
            X_scaled = self.scaler.transform(X)
            _, posteriors = self.model.score_samples(X_scaled)
            probs = {}
            for state, label in self.label_map.items():
                probs[label] = float(posteriors[-1][state])
            return probs
        except Exception:
            return {}


# ============= KMeans Fallback =============

class KMeansRegimeClassifier:
    """
    KMeans fallback when hmmlearn not available.
    Clusters 90d daily observations into 3 regimes.
    """

    FEATURES = ["returns", "volatility", "rsi", "bb_width", "momentum"]

    def __init__(self):
        self.model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.label_map = {}

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.FEATURES].values
        X_scaled = self.scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)
        self._map_labels(df, labels)
        return labels

    def _map_labels(self, df: pd.DataFrame, labels: np.ndarray):
        cluster_stats = {}
        for c in range(3):
            mask = labels == c
            if mask.sum() == 0:
                continue
            cluster_stats[c] = {
                "vol":     df["volatility"].values[mask].mean(),
                "returns": df["returns"].values[mask].mean(),
            }
        sorted_by_vol = sorted(cluster_stats.keys(), key=lambda c: cluster_stats[c]["vol"])
        if len(sorted_by_vol) == 3:
            low, mid, high = sorted_by_vol
            self.label_map = {low: "trend", mid: "revert", high: "volatile"}
        else:
            for i, c in enumerate(sorted_by_vol):
                self.label_map[c] = ["trend", "volatile"][min(i, 1)]

    def get_current_regime(self, labels: np.ndarray) -> str:
        return self.label_map.get(int(labels[-1]), "unknown")

    def get_regime_probabilities(self, df: pd.DataFrame) -> Dict[str, float]:
        return {}


# ============= Main Regime Detector =============

class RegimeDetector:
    """
    Main interface — uses HMM if available, falls back to KMeans.
    Call detect(ticker) to get full regime context for LLM prompt.
    """

    def __init__(self):
        self.use_hmm = HMM_AVAILABLE

    def detect(self, ticker: str) -> Dict:
        """
        Full regime detection pipeline.
        Returns dict ready to inject into LLM prompt.
        """
        ticker = ticker.upper()

        # Fetch + compute features
        df = compute_regime_features(ticker)
        if df is None or df.empty:
            return self._fallback(ticker, "insufficient data")

        # Classify
        try:
            if self.use_hmm:
                clf = HMMRegimeClassifier()
            else:
                clf = KMeansRegimeClassifier()

            states = clf.fit_predict(df)
            current_regime = clf.get_current_regime(states)
            probs = clf.get_regime_probabilities(df)

            # Regime distribution over last 90 days
            unique, counts = np.unique(states, return_counts=True)
            regime_dist = {}
            for state, count in zip(unique, counts):
                label = clf.label_map.get(int(state), "unknown")
                regime_dist[label] = int(count)

            # Recent trend stats
            recent = df.tail(10)
            avg_vol    = float(df["volatility"].mean() * 100)
            recent_vol = float(recent["volatility"].mean() * 100)
            momentum   = float(df["momentum"].iloc[-1] * 100)
            rsi        = float(df["rsi"].iloc[-1])

            return {
                "ticker":          ticker,
                "current_regime":  current_regime,
                "regime_probs":    probs,
                "regime_dist_90d": regime_dist,
                "stats": {
                    "avg_volatility_pct":    round(avg_vol, 3),
                    "recent_volatility_pct": round(recent_vol, 3),
                    "momentum_10d_pct":      round(momentum, 3),
                    "rsi":                   round(rsi, 2),
                },
                "method":    "HMM" if self.use_hmm else "KMeans",
                "days_used": len(df),
                "llm_context": self._build_llm_context(
                    ticker, current_regime, probs, momentum, rsi, avg_vol
                ),
                "error": None
            }

        except Exception as e:
            return self._fallback(ticker, str(e))

    def _build_llm_context(
        self, ticker: str, regime: str,
        probs: Dict, momentum: float, rsi: float, volatility: float
    ) -> str:
        """
        Builds a compact string injected into the LLM prompt.
        Tells the LLM what market regime the stock is in.
        """
        regime_descriptions = {
            "trend":    "trending (directional momentum, low noise)",
            "revert":   "mean-reverting (oscillating around a level, range-bound)",
            "volatile": "volatile (high uncertainty, large swings, elevated risk)",
        }
        desc = regime_descriptions.get(regime, regime)

        prob_str = ""
        if probs:
            prob_str = " | ".join(
                f"{k}: {v:.0%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])
            )
            prob_str = f" (regime probs: {prob_str})"

        return (
            f"MARKET REGIME [{ticker}]: {regime.upper()} — {desc}{prob_str}. "
            f"10-day momentum: {momentum:+.1f}%, RSI: {rsi:.0f}, "
            f"avg 90d volatility: {volatility:.2f}%/day. "
            f"Adjust confidence and signal strength accordingly."
        )

    def _fallback(self, ticker: str, reason: str) -> Dict:
        return {
            "ticker":         ticker,
            "current_regime": "unknown",
            "regime_probs":   {},
            "stats":          {},
            "llm_context":    f"MARKET REGIME [{ticker}]: unavailable ({reason}).",
            "error":          reason
        }


# ============= Example Usage =============

if __name__ == "__main__":
    detector = RegimeDetector()
    result = detector.detect("NVDA")

    print(f"\n=== Regime Detection: {result['ticker']} ===")
    print(f"Current regime:  {result['current_regime'].upper()}")
    print(f"Method:          {result.get('method', 'N/A')}")
    print(f"Days analyzed:   {result.get('days_used', 0)}")
    print(f"\nStats:")
    for k, v in result.get("stats", {}).items():
        print(f"  {k}: {v}")
    print(f"\nRegime distribution (90d): {result.get('regime_dist_90d', {})}")
    print(f"\nLLM context string:\n  {result['llm_context']}")
