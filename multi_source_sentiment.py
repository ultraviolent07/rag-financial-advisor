"""
Multi-Source Sentiment Analysis Engine
All sources are FREE — no API keys required except optional Reddit (PRAW).

Free sources used:
  - yfinance news (built-in, no key)
  - RSS feeds: Reuters, SeekingAlpha, Yahoo Finance
  - yfinance options chain (IV proxy)
  - yfinance analyst recommendations
  - FinBERT for NLP scoring (free HuggingFace model)
"""

from typing import Dict, List, Optional
from transformers import pipeline
import numpy as np
import yfinance as yf
import feedparser
import re
from enum import Enum
from collections import Counter
from datetime import datetime, timedelta

# ============= Free RSS Feed URLs =============

RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "yahoo_finance":    "https://finance.yahoo.com/news/rssindex",
    "seeking_alpha":    "https://seekingalpha.com/market_currents.xml",
}


# ============= FinBERT Loader (singleton) =============

_finbert = None

def get_finbert():
    """Load FinBERT once, reuse across calls"""
    global _finbert
    if _finbert is None:
        try:
            _finbert = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512
            )
            print("FinBERT loaded")
        except Exception as e:
            print(f"FinBERT failed ({e}), falling back to zero-shot")
            _finbert = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
    return _finbert


# ============= Free Data Fetchers =============

class FreeNewsFetcher:
    """
    Fetch financial news with zero API keys.
    Sources: yfinance built-in news + RSS feeds filtered by ticker.
    """

    @staticmethod
    def fetch_yfinance_news(ticker: str, max_articles: int = 10) -> List[str]:
        """yfinance has built-in news — completely free"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            headlines = []
            for item in news[:max_articles]:
                title = item.get("title", "")
                summary = item.get("summary", "")
                if title:
                    headlines.append(f"{title}. {summary}".strip())
            return headlines
        except Exception as e:
            print(f"yfinance news error for {ticker}: {e}")
            return []

    @staticmethod
    def fetch_rss_news(ticker: str, company_name: str = "", max_articles: int = 10) -> List[str]:
        """
        Pull from free RSS feeds, filter articles mentioning ticker or company.
        No API key needed.
        """
        headlines = []
        search_terms = [ticker.upper(), company_name.upper()] if company_name else [ticker.upper()]

        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:  # check first 30 entries per feed
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    combined = f"{title} {summary}".upper()

                    if any(term in combined for term in search_terms):
                        headlines.append(f"{title}. {summary[:200]}".strip())

                    if len(headlines) >= max_articles:
                        break
            except Exception as e:
                print(f"RSS feed {feed_name} error: {e}")

        return headlines[:max_articles]

    @staticmethod
    def fetch_all(ticker: str, company_name: str = "") -> List[str]:
        """Combine yfinance + RSS, deduplicate"""
        yf_news = FreeNewsFetcher.fetch_yfinance_news(ticker)
        rss_news = FreeNewsFetcher.fetch_rss_news(ticker, company_name)
        all_news = yf_news + rss_news

        # Deduplicate by first 60 chars
        seen = set()
        unique = []
        for article in all_news:
            key = article[:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(article)

        return unique


class FreeMarketDataFetcher:
    """
    Fetch IV proxy and analyst data from yfinance — completely free.
    """

    @staticmethod
    def fetch_iv_proxy(ticker: str) -> Optional[Dict]:
        """
        Approximate IV from options chain via yfinance.
        Uses nearest expiry ATM options implied vol as IV proxy.
        """
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get("currentPrice") or stock.info.get("regularMarketPrice")
            if not current_price:
                hist = stock.history(period="1d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None

            if not current_price:
                return None

            expirations = stock.options
            if not expirations:
                return None

            # Use nearest expiry
            nearest = expirations[0]
            chain = stock.option_chain(nearest)

            # Find ATM calls
            calls = chain.calls
            calls = calls[calls["impliedVolatility"] > 0]
            if calls.empty:
                return None

            # ATM = strike closest to current price
            calls["dist"] = abs(calls["strike"] - current_price)
            atm = calls.nsmallest(3, "dist")
            current_iv = float(atm["impliedVolatility"].mean()) * 100  # to percentage

            # Get 30-day historical vol as baseline
            hist = stock.history(period="30d")
            if len(hist) > 5:
                returns = hist["Close"].pct_change().dropna()
                hist_vol = float(returns.std() * np.sqrt(252) * 100)
            else:
                hist_vol = current_iv  # fallback

            return {
                "current_iv": current_iv,
                "avg_iv": hist_vol,
                "ticker": ticker
            }

        except Exception as e:
            print(f"IV fetch error for {ticker}: {e}")
            return None

    @staticmethod
    def fetch_analyst_ratings(ticker: str) -> Optional[Dict]:
        """
        Fetch analyst buy/hold/sell counts from yfinance recommendations.
        Completely free.
        """
        try:
            stock = yf.Ticker(ticker)
            recs = stock.recommendations

            if recs is None or recs.empty:
                # Fallback to info-level data
                info = stock.info
                return {
                    "buy":  info.get("recommendationMean", 3),
                    "hold": 0,
                    "sell": 0,
                    "source": "yfinance_info"
                }

            # Use last 90 days of recommendations
            cutoff = datetime.now() - timedelta(days=90)
            if hasattr(recs.index, 'tz_localize'):
                recent = recs[recs.index >= cutoff.strftime("%Y-%m-%d")]
            else:
                recent = recs.tail(20)  # fallback: last 20

            if recent.empty:
                recent = recs.tail(10)

            # Normalize column names (yfinance changed these)
            col_map = {}
            for col in recent.columns:
                cl = col.lower()
                if "strong buy" in cl or "strongbuy" in cl:
                    col_map[col] = "strong_buy"
                elif "buy" in cl:
                    col_map[col] = "buy"
                elif "hold" in cl or "neutral" in cl:
                    col_map[col] = "hold"
                elif "sell" in cl and "strong" not in cl:
                    col_map[col] = "sell"
                elif "strong sell" in cl or "strongsell" in cl:
                    col_map[col] = "strong_sell"

            recent = recent.rename(columns=col_map)

            buy  = int(recent.get("buy",  recent.get("strong_buy",  0)).sum() if "buy"  in recent else 0)
            hold = int(recent.get("hold", 0).sum() if "hold" in recent else 0)
            sell = int(recent.get("sell", recent.get("strong_sell", 0)).sum() if "sell" in recent else 0)

            return {"buy": buy, "hold": hold, "sell": sell, "source": "yfinance_recommendations"}

        except Exception as e:
            print(f"Analyst ratings error for {ticker}: {e}")
            return None


# ============= Sentiment Analyzers =============

class NewsSentimentAnalyzer:
    """Score news headlines with FinBERT"""

    def __init__(self):
        self.classifier = get_finbert()

    def analyze(self, texts: List[str], weights: List[float] = None) -> Dict:
        if not texts:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "sample_size": 0}

        if weights is None:
            weights = [1.0] * len(texts)

        sentiments, scores = [], []

        for text, weight in zip(texts, weights):
            try:
                result = self.classifier(text[:512])
                if isinstance(result, list) and result:
                    label = result[0]["label"].lower()
                    score = result[0]["score"]
                else:
                    continue

                val = 1.0 if "positive" in label else (-1.0 if "negative" in label else 0.0)
                sentiments.append(val * weight)
                scores.append(score * weight)
            except Exception:
                pass

        if not sentiments:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "sample_size": 0}

        avg = float(np.mean(sentiments))
        label = "positive" if avg > 0.2 else ("negative" if avg < -0.2 else "neutral")

        return {
            "source": "news_finbert",
            "sentiment": label,
            "score": avg,
            "confidence": float(np.mean(scores)),
            "sample_size": len(texts)
        }


class OptionsMarketSentiment:
    """IV-based fear/greed proxy"""

    @staticmethod
    def from_iv(iv_data: Dict) -> Dict:
        current = iv_data.get("current_iv", 0)
        avg = iv_data.get("avg_iv", current or 1)
        ratio = current / avg if avg > 0 else 1.0

        if ratio > 1.2:
            sentiment, score = "negative", -min(ratio - 1, 1.0)
        elif ratio < 0.8:
            sentiment, score = "positive", min(1 - ratio, 1.0)
        else:
            sentiment, score = "neutral", 0.0

        return {
            "source": "options_iv",
            "sentiment": sentiment,
            "score": float(score),
            "confidence": abs(float(score)),
            "iv_ratio": float(ratio),
            "current_iv": current,
            "hist_vol": avg
        }


class AnalystSentiment:
    """Analyst ratings aggregator"""

    @staticmethod
    def from_ratings(buy: int, hold: int, sell: int) -> Dict:
        total = buy + hold + sell
        if total == 0:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

        score = (buy - sell) / total
        label = "positive" if score > 0.3 else ("negative" if score < -0.3 else "neutral")

        return {
            "source": "analyst",
            "sentiment": label,
            "score": float(score),
            "confidence": float((buy + sell) / total),
            "breakdown": {
                "buy": buy, "hold": hold, "sell": sell,
                "buy_pct": round(buy / total, 2) if total else 0
            }
        }


# ============= Ensemble Engine =============

class EnsembleSentimentAnalyzer:
    """
    Auto-fetches all data from free sources, runs FinBERT,
    fuses scores with weighted ensemble.
    """

    def __init__(self):
        self.news_analyzer    = NewsSentimentAnalyzer()
        self.options_analyzer = OptionsMarketSentiment()
        self.analyst_analyzer = AnalystSentiment()
        self.news_fetcher     = FreeNewsFetcher()
        self.market_fetcher   = FreeMarketDataFetcher()

        self.weights = {
            "news":    0.35,
            "options": 0.25,
            "analyst": 0.40,
        }

    def analyze_ticker(self, ticker: str, company_name: str = "") -> Dict:
        """
        Full auto pipeline — just pass a ticker.
        Fetches news, IV, analyst ratings → runs ensemble → returns result.
        """
        ticker = ticker.upper()
        results = {}
        scores  = []
        confs   = []

        # 1. News (yfinance + RSS → FinBERT)
        print(f"Fetching news for {ticker}...")
        news = self.news_fetcher.fetch_all(ticker, company_name)
        if news:
            nr = self.news_analyzer.analyze(news)
            results["news"] = nr
            scores.append(nr["score"] * self.weights["news"])
            confs.append(nr["confidence"])
        else:
            print(f"  No news found for {ticker}")

        # 2. Options IV proxy
        print(f"Fetching IV for {ticker}...")
        iv = self.market_fetcher.fetch_iv_proxy(ticker)
        if iv:
            or_ = self.options_analyzer.from_iv(iv)
            results["options"] = or_
            scores.append(or_["score"] * self.weights["options"])
            confs.append(or_["confidence"])
        else:
            print(f"  No options data for {ticker}")

        # 3. Analyst ratings
        print(f"Fetching analyst ratings for {ticker}...")
        analyst = self.market_fetcher.fetch_analyst_ratings(ticker)
        if analyst:
            ar = self.analyst_analyzer.from_ratings(
                analyst.get("buy", 0),
                analyst.get("hold", 0),
                analyst.get("sell", 0)
            )
            results["analyst"] = ar
            scores.append(ar["score"] * self.weights["analyst"])
            confs.append(ar["confidence"])

        # Ensemble
        overall_score = float(sum(scores)) if scores else 0.0
        overall_conf  = float(np.mean(confs)) if confs else 0.0
        overall_label = "positive" if overall_score > 0.2 else (
                        "negative" if overall_score < -0.2 else "neutral")

        source_labels = [r.get("sentiment", "neutral") for r in results.values()]
        agreement = self._agreement(source_labels)
        signals   = self._signals(results, overall_score)

        return {
            "ticker":             ticker,
            "overall_sentiment":  overall_label,
            "overall_score":      round(overall_score, 3),
            "confidence":         round(overall_conf, 3),
            "breakdown":          results,
            "agreement":          agreement,
            "signals":            signals,
            "risk_level":         "high" if overall_label == "negative" and overall_conf > 0.6
                                  else ("low" if overall_label == "positive" else "medium"),
            "recommendation":     self._recommend(overall_label, agreement),
            "sources_analyzed":   len(results),
            "news_count":         len(news) if news else 0,
        }

    # Keep old interface for backward compat with backend_main.py
    def analyze_comprehensive(self,
                               ticker: str,
                               news_texts: List[str] = None,
                               iv_data: Dict = None,
                               analyst_data: Dict = None,
                               social_sentiment: Dict = None) -> Dict:
        """
        Legacy interface — pass data manually OR just call analyze_ticker().
        If data not passed, auto-fetches from free sources.
        """
        if news_texts is None and iv_data is None and analyst_data is None:
            return self.analyze_ticker(ticker)

        # Manual path (original behavior)
        results, scores, confs = {}, [], []

        if news_texts:
            nr = self.news_analyzer.analyze(news_texts)
            results["news"] = nr
            scores.append(nr.get("score", 0) * self.weights["news"])
            confs.append(nr.get("confidence", 0))

        if iv_data:
            or_ = self.options_analyzer.from_iv(iv_data)
            results["options"] = or_
            scores.append(or_.get("score", 0) * self.weights["options"])
            confs.append(or_.get("confidence", 0))

        if analyst_data:
            ar = self.analyst_analyzer.from_ratings(
                analyst_data.get("buy", 0),
                analyst_data.get("hold", 0),
                analyst_data.get("sell", 0)
            )
            results["analyst"] = ar
            scores.append(ar.get("score", 0) * self.weights["analyst"])
            confs.append(ar.get("confidence", 0))

        overall_score = float(sum(scores)) if scores else 0.0
        overall_conf  = float(np.mean(confs)) if confs else 0.0
        overall_label = "positive" if overall_score > 0.2 else (
                        "negative" if overall_score < -0.2 else "neutral")

        source_labels = [r.get("sentiment", "neutral") for r in results.values()]
        return {
            "ticker": ticker,
            "overall_sentiment": overall_label,
            "overall_score": round(overall_score, 3),
            "confidence": round(overall_conf, 3),
            "breakdown": results,
            "agreement": self._agreement(source_labels),
            "signals": self._signals(results, overall_score),
            "risk_level": "medium",
            "recommendation": self._recommend(overall_label, self._agreement(source_labels)),
            "sources_analyzed": len(results),
        }

    @staticmethod
    def _agreement(sentiments: List[str]) -> str:
        if not sentiments:
            return "unknown"
        counts = Counter(sentiments)
        pct = max(counts.values()) / len(sentiments)
        return "strong" if pct > 0.75 else ("moderate" if pct > 0.5 else "weak")

    @staticmethod
    def _signals(results: Dict, overall_score: float) -> List[str]:
        signals = []
        if "news" in results:
            s = results["news"].get("sentiment")
            n = results["news"].get("sample_size", 0)
            if s == "positive" and n > 3:
                signals.append("Positive news coverage (FinBERT)")
            elif s == "negative" and n > 3:
                signals.append("Negative news coverage (FinBERT)")

        if "options" in results:
            sc = results["options"].get("score", 0)
            ratio = results["options"].get("iv_ratio", 1)
            if sc < -0.4:
                signals.append(f"Elevated IV ({ratio:.1f}x hist) — market fear")
            elif sc > 0.4:
                signals.append("Low IV — calm/bullish options market")

        if "analyst" in results:
            bp = results["analyst"].get("breakdown", {}).get("buy_pct", 0)
            if bp > 0.7:
                signals.append("Analyst consensus: bullish")
            elif bp < 0.3:
                signals.append("Analyst consensus: bearish")

        if overall_score > 0.6:
            signals.append("Strong buy signal (multi-source)")
        elif overall_score < -0.6:
            signals.append("Strong sell signal (multi-source)")

        return signals

    @staticmethod
    def _recommend(sentiment: str, agreement: str) -> str:
        matrix = {
            ("positive", "strong"):   "STRONG BUY",
            ("positive", "moderate"): "BUY",
            ("positive", "weak"):     "HOLD",
            ("negative", "strong"):   "STRONG SELL",
            ("negative", "moderate"): "SELL",
            ("negative", "weak"):     "HOLD",
        }
        return matrix.get((sentiment, agreement), "HOLD")


# ============= Example Usage =============

if __name__ == "__main__":
    analyzer = EnsembleSentimentAnalyzer()

    print("\n=== Auto-fetch sentiment for AAPL ===")
    result = analyzer.analyze_ticker("AAPL", company_name="Apple")

    print(f"\nTicker:         {result['ticker']}")
    print(f"Sentiment:      {result['overall_sentiment'].upper()}")
    print(f"Score:          {result['overall_score']:.3f}")
    print(f"Confidence:     {result['confidence']:.3f}")
    print(f"Agreement:      {result['agreement']}")
    print(f"Risk:           {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"News articles:  {result['news_count']}")
    print("\nSignals:")
    for s in result["signals"]:
        print(f"  {s}")
    print("\nSource breakdown:")
    for src, data in result["breakdown"].items():
        print(f"  {src}: {data.get('sentiment')} (score: {data.get('score', 0):.3f})")