"""
Finance module — Tier 3 upgrade
Wires regime detection into StockAnalysisService.
analyze_stock() now returns regime label + LLM context string.
"""
from typing import Dict, List
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from regime_detection import RegimeDetector
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    print("⚠️  regime_detection not found — regime analysis disabled")


# ============= Stock Data Fetcher =============

class StockDataFetcher:
    """Fetch real-time and historical stock data"""

    @staticmethod
    def get_current_price(ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            data  = stock.history(period="1d")
            if data.empty:
                return {"error": f"Ticker {ticker} not found"}
            return {
                "ticker":    ticker,
                "price":     float(data["Close"].iloc[-1]),
                "currency":  "USD",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_historical_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
        try:
            return yf.download(ticker, period=period, progress=False)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_company_info(ticker: str) -> Dict:
        try:
            info = yf.Ticker(ticker).info
            return {
                "name":            info.get("longName", ""),
                "sector":          info.get("sector", ""),
                "industry":        info.get("industry", ""),
                "market_cap":      info.get("marketCap", 0),
                "pe_ratio":        info.get("trailingPE", 0),
                "forward_pe":      info.get("forwardPE", 0),
                "peg_ratio":       info.get("pegRatio", 0),
                "dividend_yield":  info.get("dividendYield", 0),
                "52_week_high":    info.get("fiftyTwoWeekHigh", 0),
                "52_week_low":     info.get("fiftyTwoWeekLow", 0),
                "analyst_target":  info.get("targetMeanPrice", 0),
            }
        except Exception as e:
            return {"error": str(e)}


# ============= Financial Metrics =============

class FinancialMetrics:

    @staticmethod
    def calculate_returns(prices: pd.Series) -> Dict:
        daily_returns = prices.pct_change()
        return {
            "total_return":     float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
            "avg_daily_return": float(daily_returns.mean() * 100),
            "volatility":       float(daily_returns.std() * 100),
            "sharpe_ratio":     float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
                                if daily_returns.std() > 0 else 0.0
        }

    @staticmethod
    def calculate_moving_averages(prices: pd.Series) -> Dict:
        ma_20  = prices.rolling(20).mean().iloc[-1]
        ma_50  = prices.rolling(50).mean().iloc[-1]
        ma_200 = prices.rolling(200).mean().iloc[-1]
        last   = float(prices.iloc[-1])
        return {
            "ma_20":          float(ma_20)  if not pd.isna(ma_20)  else None,
            "ma_50":          float(ma_50)  if not pd.isna(ma_50)  else None,
            "ma_200":         float(ma_200) if not pd.isna(ma_200) else None,
            "price_to_ma20":  round(last / float(ma_20),  3) if not pd.isna(ma_20)  else None,
            "price_to_ma50":  round(last / float(ma_50),  3) if not pd.isna(ma_50)  else None,
            "price_to_ma200": round(last / float(ma_200), 3) if not pd.isna(ma_200) else None,
        }

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain  = delta.where(delta > 0, 0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> Dict:
        sma   = prices.rolling(period).mean().iloc[-1]
        std   = prices.rolling(period).std().iloc[-1]
        last  = float(prices.iloc[-1])
        upper = float(sma + 2 * std)
        lower = float(sma - 2 * std)
        return {
            "upper": round(upper, 2),
            "middle": round(float(sma), 2),
            "lower": round(lower, 2),
            "pct_b": round((last - lower) / (upper - lower), 3) if (upper - lower) > 0 else 0.5
        }


# ============= Stock Analysis Service =============

class StockAnalysisService:
    """
    Tier 3: Comprehensive stock analysis with regime detection.
    Returns regime label + llm_context string ready for LLM prompt injection.
    """

    def __init__(self):
        self.fetcher  = StockDataFetcher()
        self.metrics  = FinancialMetrics()
        self.regime_detector = RegimeDetector() if REGIME_AVAILABLE else None

    def analyze_stock(self, ticker: str, analysis_type: str = "full") -> Dict:
        """
        Full analysis: technicals + fundamentals + regime detection.
        Returns technicals_summary string for LLM prompt injection.
        """
        analysis = {
            "ticker":        ticker,
            "timestamp":     datetime.now().isoformat(),
            "analysis_type": analysis_type
        }

        # Price
        price_info = self.fetcher.get_current_price(ticker)
        if "error" in price_info:
            analysis["error"] = price_info["error"]
            return analysis
        analysis["price"] = price_info["price"]

        # Historical data
        hist_data = self.fetcher.get_historical_data(ticker, period="1y")
        if hist_data.empty:
            analysis["error"] = f"No data for {ticker}"
            return analysis

        prices = hist_data["Close"].squeeze()

        # Technical analysis
        if analysis_type in ["full", "technical"]:
            returns = self.metrics.calculate_returns(prices)
            mas     = self.metrics.calculate_moving_averages(prices)
            rsi     = self.metrics.calculate_rsi(prices)
            bb      = self.metrics.calculate_bollinger_bands(prices)

            signal = self._generate_signal(returns, mas, rsi)

            analysis["technical"] = {
                "returns":          returns,
                "moving_averages":  mas,
                "rsi":              rsi,
                "bollinger_bands":  bb,
                "signal":           signal
            }

            # Build technicals summary string for LLM
            analysis["technicals_summary"] = (
                f"Price: ${analysis['price']:.2f} | "
                f"RSI: {rsi:.1f} | "
                f"1Y return: {returns['total_return']:.1f}% | "
                f"Volatility: {returns['volatility']:.2f}%/day | "
                f"vs MA50: {(mas.get('price_to_ma50') or 1)*100-100:+.1f}% | "
                f"Signal: {signal}"
            )

        # Fundamental analysis
        if analysis_type in ["full", "fundamental"]:
            analysis["fundamental"] = self.fetcher.get_company_info(ticker)

        # Regime detection (Tier 3)
        if analysis_type in ["full", "regime"] and self.regime_detector:
            print(f"  🔍 Running regime detection for {ticker}...")
            regime_result = self.regime_detector.detect(ticker)
            analysis["regime"] = {
                "current":     regime_result["current_regime"],
                "probs":       regime_result.get("regime_probs", {}),
                "dist_90d":    regime_result.get("regime_dist_90d", {}),
                "stats":       regime_result.get("stats", {}),
                "method":      regime_result.get("method", "N/A"),
                "error":       regime_result.get("error")
            }
            # LLM-ready context string
            analysis["regime_context"] = regime_result.get("llm_context", "")
        else:
            analysis["regime"] = {"current": "unknown", "error": "regime detection unavailable"}
            analysis["regime_context"] = ""

        return analysis

    def get_technicals_summary(self, ticker: str) -> str:
        """Quick technicals string for LLM prompt — used by groq_integration"""
        result = self.analyze_stock(ticker, analysis_type="technical")
        return result.get("technicals_summary", f"Technicals unavailable for {ticker}")

    @staticmethod
    def _generate_signal(returns: Dict, mas: Dict, rsi: float) -> str:
        signals = []
        if rsi > 70:
            signals.append("OVERBOUGHT (RSI>70)")
        elif rsi < 30:
            signals.append("OVERSOLD (RSI<30)")

        p50 = mas.get("price_to_ma50")
        if p50 and p50 > 1.05:
            signals.append("Above MA50 ↑")
        elif p50 and p50 < 0.95:
            signals.append("Below MA50 ↓")

        ret = returns.get("total_return", 0)
        if ret > 20:
            signals.append(f"Strong 1Y return +{ret:.0f}%")
        elif ret < -15:
            signals.append(f"Weak 1Y return {ret:.0f}%")

        return " | ".join(signals) if signals else "NEUTRAL"


# ============= Example Usage =============

if __name__ == "__main__":
    service = StockAnalysisService()

    print("=== Full Stock Analysis (NVDA) ===")
    result = service.analyze_stock("NVDA", analysis_type="full")

    print(f"Ticker:  {result['ticker']}")
    print(f"Price:   ${result.get('price', 0):.2f}")

    if "technical" in result:
        t = result["technical"]
        print(f"RSI:     {t['rsi']:.1f}")
        print(f"Signal:  {t['signal']}")

    if "regime" in result:
        r = result["regime"]
        print(f"Regime:  {r.get('current', 'N/A').upper()}")
        print(f"Method:  {r.get('method', 'N/A')}")

    print(f"\nLLM regime context:\n  {result.get('regime_context', 'N/A')}")
    print(f"\nLLM technicals summary:\n  {result.get('technicals_summary', 'N/A')}")
