AI-Powered Financial Research & Trading Advisor
A production-grade RAG system for financial document analysis, combining advanced retrieval, market regime detection, and multi-source sentiment analysis into a single investment research pipeline.

Architecture
User Query + PDF Upload
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Three Parallel Async Pipelines          │
├─────────────────┬──────────────────┬────────────────┤
│   Advanced RAG  │ Regime Detection │Sentiment Fusion│
│                 │                  │                │
│ Query expansion │ yfinance 90d data│ yfinance news  │
│ Hybrid retrieval│ RSI, ATR, BB     │ RSS feeds      │
│ BM25 + ChromaDB │ HMM / KMeans     │ FinBERT scoring│
│ RRF fusion      │ trend/revert/    │ Options IV     │
│ Parent-child    │ volatile label   │ Analyst ratings│
│ chunking        │                  │                │
└─────────────────┴──────────────────┴────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│           LLM Reasoning — Decomposed 3-Call Chain    │
│  Call 1 (fast):  classify regime + sentiment align   │
│  Call 2 (bal):   extract key claims from RAG docs    │
│  Call 3 (bal):   synthesise signal + chain-of-thought│
│  Conflict flag → forced uncertainty hedge in output  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Response                    │
│  signal · regime · sentiment · conflict · sources   │
└─────────────────────────────────────────────────────┘
Key Features
Tier 2 — Advanced RAG
Parent-child chunking — indexes 150-token child chunks for precise retrieval, returns 600-token parent chunks to the LLM for full context. Eliminates hallucination caused by context-starved chunks.
Hybrid BM25 + ChromaDB retrieval — semantic search catches meaning, BM25 catches exact matches (ticker names, dates, "Q3 2024"). Run in parallel on every query.
Reciprocal Rank Fusion (RRF) — fuses BM25 and semantic ranked lists mathematically without a learned model. Highest-ROI RAG improvement.
Query expansion — one fast LLM call generates 3 reformulations before retrieval. "Is NVDA overvalued?" also searches "stretched valuations", "high P/E multiple".
Tier 3 — Market Intelligence
HMM regime detection — Gaussian Hidden Markov Model trained on 90 days of yfinance price data. Labels market as trend / revert / volatile using returns, RSI, Bollinger Band width, volume z-score, momentum. Falls back to KMeans if hmmlearn unavailable.
FinBERT sentiment — financial-domain transformer (ProsusAI/finbert) replaces generic sentiment. Scores news headlines specifically trained on financial text.
Free multi-source sentiment — yfinance built-in news + Reuters/Yahoo Finance/SeekingAlpha RSS feeds + options IV proxy + analyst recommendation aggregation. Zero external API keys required.
3-call decomposed Groq chain — adaptive depth routing: simple queries go fast path (1 call), deep research triggers full 3-call chain: classify → extract claims → synthesise with chain-of-thought. Conflict detection forces uncertainty hedge in output.
Tech Stack
Layer	Technology
Backend	FastAPI, Python 3.12
LLM Inference	Groq (LLaMA 3.3 70B, LLaMA 3.1 8B) — free tier
Vector Store	ChromaDB (PersistentClient)
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Keyword Search	rank-bm25
Sentiment NLP	ProsusAI/FinBERT (HuggingFace)
Regime Detection	hmmlearn (Gaussian HMM) + scikit-learn (KMeans)
Market Data	yfinance (free, no API key)
News	feedparser + yfinance news (free, no API key)
PDF Parsing	pdfplumber (batch streaming, 200-page cap)
Project Structure
trading_advisor_updated/
├── backend_main.py          # FastAPI app, all endpoints
├── rag_pipeline.py          # Tier 2 RAG: hybrid retrieval, RRF, parent-child
├── groq_integration.py      # 3-call decomposed LLM chain
├── finance_module.py        # Technicals, fundamentals, regime wiring
├── regime_detection.py      # HMM/KMeans market regime classifier
├── multi_source_sentiment.py# FinBERT + free data source fetchers
├── pdf_parser.py            # Streaming PDF parser for long documents
├── config.py                # Environment config
├── utils.py                 # Caching, rate limiting, validation
├── verify_setup.py          # Pre-flight checks
├── requirements.txt
├── .env                     # GROQ_API_KEY (never commit)
├── data/
│   ├── chroma_db/           # ChromaDB persisted vector store
│   └── uploads/             # Uploaded PDFs
└── logs/
Setup
Prerequisites
Python 3.10+
Free Groq API key from console.groq.com
Installation
# Clone and enter directory
git clone https://github.com/yourusername/trading-advisor.git
cd trading-advisor

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=gsk_your_key_here" > .env
echo "API_PORT=8000" >> .env

# Verify setup
python verify_setup.py

# Run backend
uvicorn backend_main:app --reload
Open http://localhost:8000/docs to access the interactive API.

API Endpoints
Method	Endpoint	Description
GET	/health	System health + component status
POST	/chat	RAG chat with optional deep research mode
POST	/documents/upload	Upload PDF/DOCX for RAG indexing
GET	/documents	List uploaded documents
DELETE	/documents/{file_id}	Remove document
POST	/analysis/stock	Full analysis: technicals + regime + sentiment
GET	/analysis/sentiment/{ticker}	Multi-source sentiment only
GET	/analysis/watchlist	Batch analysis for AAPL, MSFT, TSLA, GOOGL, NVDA
Chat endpoint — deep research mode
POST /chat
{
  "query": "Should I invest in Apple given current market conditions?",
  "ticker": "AAPL",
  "deep_research": true
}
Response includes signal, conflict, calls_made, and chain-of-thought reasoning citing document sources.

Example Output
POST /analysis/stock — AAPL

{
  "price": 283.78,
  "regime": {
    "current": "revert",
    "method": "HMM",
    "dist_90d": {"volatile": 33, "revert": 29, "trend": 14}
  },
  "regime_context": "MARKET REGIME [AAPL]: REVERT — mean-reverting, range-bound. 10-day momentum: -4.0%, RSI: 33, avg 90d volatility: 1.43%/day.",
  "technical": {
    "rsi": 33.48,
    "signal": "Strong 1Y return +42%"
  },
  "sentiment": 0.284
}
POST /chat — deep research

{
  "signal": "bullish",
  "conflict": false,
  "calls_made": 3,
  "confidence": 0.9,
  "response": "Based on Q2 2025 results ($95,359M revenue, gross margin $44,867M) and current RSI 33 (oversold), the mean-reverting regime suggests a potential entry point..."
}
How It Works
RAG Pipeline
PDF uploaded → streamed in 20-page batches → parent-child chunked (150/600 tokens)
Child chunks indexed in ChromaDB + BM25 simultaneously
Query arrives → expanded to 3 reformulations via LLM
All 4 queries run through both ChromaDB (semantic) and BM25 (keyword)
Results fused via Reciprocal Rank Fusion → top-5 parent chunks returned to LLM
Regime Detection
90 days of OHLCV data fetched from yfinance
Six features computed: returns, volatility, RSI, Bollinger Band width, volume z-score, momentum
Gaussian HMM with 3 hidden states fit on the feature matrix
States mapped to trend/revert/volatile by volatility and return characteristics
Current-day posterior probability returned alongside Viterbi state label
3-Call Chain (deep research)
Call 1 (LLaMA 8B, fast): classify regime-sentiment alignment → {signal, conflict, alignment}
Call 2 (LLaMA 70B): extract factual claims from RAG context → {claims, data_points, gaps}
Call 3 (LLaMA 70B): synthesise all context into chain-of-thought analysis. If conflict detected → forced uncertainty hedge injected into prompt.
Resume Highlights
Built Tier-2 RAG with parent-child chunking, hybrid BM25+ChromaDB retrieval, RRF fusion, and LLM query expansion over SEC filings and earnings documents.
Engineered HMM-based market regime classifier (trend/revert/volatile) on 90-day yfinance features; integrated FinBERT sentiment ensemble with zero external API dependencies.
Designed adaptive 3-call Groq LLM chain with conflict detection, uncertainty hedging, and chain-of-thought synthesis; served via FastAPI with caching, rate limiting, and streaming support.
Notes
All data sources are free — yfinance, HuggingFace models, RSS feeds, Groq free tier
No OpenAI, NewsAPI, or paid services required
ChromaDB persists locally — documents survive server restarts
Rate limiting and caching built in to stay within Groq free tier limits
License
MIT
