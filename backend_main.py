"""
FastAPI Backend - AI Trading Advisor
Tier 3 update: regime detection wired in, deep_research flag added to chat
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
import uuid
import time
from datetime import datetime

from config import (
    CORS_ORIGINS, API_PORT, DEBUG, ENVIRONMENT,
    GROQ_API_KEY, RATE_LIMIT_ENABLED, RATE_LIMIT_CALLS_PER_MINUTE,
    UPLOAD_DIR, CACHE_ENABLED, CACHE_TTL, print_config, validate_config
)
from utils import (
    setup_logger, ResponseFormatter, Validator, ErrorResponse,
    RateLimiter, SimpleCache, DocumentException, RAGException, LLMException,
    FinanceException, check_initialization
)

from rag_pipeline import initialize_rag_pipeline, RAGRetriever
from groq_integration import GroqLLMChain, GroqRAGChain
from finance_module import StockAnalysisService
from multi_source_sentiment import EnsembleSentimentAnalyzer
from pdf_parser import DocumentLoader

try:
    from regime_detection import RegimeDetector
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False

logger = setup_logger(__name__)

# ============= App =============

app = FastAPI(
    title="AI Trading Advisor",
    description="RAG-powered financial research and trading insights",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Global State =============

rag_retriever      = None
groq_llm           = None
rag_chain          = None
stock_analyzer     = None
sentiment_analyzer = None
document_loader    = None
regime_detector    = None

response_cache = SimpleCache(ttl=CACHE_TTL) if CACHE_ENABLED else None
rate_limiter   = RateLimiter(
    max_calls=RATE_LIMIT_CALLS_PER_MINUTE, time_window=60
) if RATE_LIMIT_ENABLED else None

# ============= Pydantic Models =============

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    conversation_history: List[dict] = Field(default_factory=list)
    deep_research: bool = Field(default=False)          # NEW: triggers 3-call chain
    ticker: Optional[str] = Field(default=None)         # NEW: optional for regime context

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    confidence: float
    analysis_time: float
    signal: Optional[str] = None                        # NEW
    conflict: Optional[bool] = None                     # NEW
    calls_made: Optional[int] = None                    # NEW

class DocumentUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    chunks_created: int
    upload_time: str

class StockAnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    analysis_type: str = Field(default="full")

class StockAnalysisResponse(BaseModel):
    ticker: str
    price: float
    analysis: str
    sentiment: float
    technical: Dict
    fundamental: Dict
    regime: Optional[Dict] = None                       # NEW
    regime_context: Optional[str] = None                # NEW
    technicals_summary: Optional[str] = None            # NEW
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    components: Dict[str, bool]

# ============= Startup =============

@app.on_event("startup")
async def startup_event():
    global rag_retriever, groq_llm, rag_chain, stock_analyzer
    global sentiment_analyzer, document_loader, regime_detector

    try:
        if not validate_config():
            raise Exception("Configuration validation failed")

        logger.info("Starting up Trading Advisor...")
        print_config()

        logger.info("Initializing RAG pipeline...")
        rag_retriever = initialize_rag_pipeline()

        logger.info("Initializing Groq LLM...")
        groq_llm = GroqLLMChain(api_key=GROQ_API_KEY)

        logger.info("Initializing RAG Chain...")
        rag_chain = GroqRAGChain(rag_retriever, groq_llm)

        logger.info("Initializing Finance Services...")
        stock_analyzer     = StockAnalysisService()
        sentiment_analyzer = EnsembleSentimentAnalyzer()

        logger.info("Initializing Document Loader...")
        document_loader = DocumentLoader(chunk_strategy="rag")

        if REGIME_AVAILABLE:
            logger.info("Initializing Regime Detector...")
            regime_detector = RegimeDetector()
            logger.info("Regime detector ready (HMM/KMeans)")
        else:
            logger.warning("Regime detector not available")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Trading Advisor...")

# ============= Health =============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    components = {
        "rag_retriever":      rag_retriever is not None,
        "groq_llm":           groq_llm is not None,
        "rag_chain":          rag_chain is not None,
        "stock_analyzer":     stock_analyzer is not None,
        "sentiment_analyzer": sentiment_analyzer is not None,
        "document_loader":    document_loader is not None,
        "regime_detector":    regime_detector is not None,
    }
    return HealthResponse(
        status="healthy" if all(components.values()) else "degraded",
        environment=ENVIRONMENT,
        version="2.0.0",
        components=components
    )

@app.get("/")
async def root():
    return ResponseFormatter.success(
        data={"name": "AI Trading Advisor", "version": "2.0.0", "docs": "/docs"},
        message="Welcome to Trading Advisor API"
    )

# ============= Chat =============

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main RAG chat endpoint.
    deep_research=true → 3-call Groq chain (classify → extract → synthesise)
    deep_research=false → adaptive fast/single-call path
    """
    try:
        Validator.validate_query(request.query)
        Validator.validate_conversation_history(request.conversation_history)

        if rate_limiter and not rate_limiter.is_allowed("chat"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Cache only for non-deep-research queries
        cache_key = f"chat:{hash(request.query)}:{request.deep_research}"
        if response_cache and not request.deep_research:
            cached = response_cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for query: {request.query[:50]}")
                return cached

        start_time = time.time()

        # Fetch regime + technicals context if ticker provided
        regime_context     = ""
        technicals_summary = ""

        if request.ticker and request.deep_research:
            try:
                ticker = request.ticker.upper()
                if regime_detector:
                    rd = regime_detector.detect(ticker)
                    regime_context = rd.get("llm_context", "")
                technicals_summary = stock_analyzer.get_technicals_summary(ticker)
            except Exception as e:
                logger.warning(f"Context fetch failed for {request.ticker}: {e}")

        # Fetch sentiment summary if ticker provided
        sentiment_summary = ""
        if request.ticker and request.deep_research:
            try:
                sent = sentiment_analyzer.analyze_ticker(request.ticker.upper())
                sentiment_summary = (
                    f"Sentiment: {sent['overall_sentiment']} "
                    f"(score: {sent['overall_score']:.2f}, "
                    f"recommendation: {sent['recommendation']})"
                )
            except Exception as e:
                logger.warning(f"Sentiment fetch failed: {e}")

        result = rag_chain.answer_question(
            query=request.query,
            history=request.conversation_history,
            regime_context=regime_context,
            sentiment_summary=sentiment_summary,
            technicals_summary=technicals_summary,
            deep_research=request.deep_research
        )

        analysis_time = time.time() - start_time
        logger.info(f"Chat query processed in {analysis_time:.2f}s "
                    f"(deep={request.deep_research}, calls={result.get('calls_made',1)})")

        response = ChatResponse(
            response=result["answer"],
            sources=result["sources"][:3],
            confidence=result.get("confidence", 0.85),
            analysis_time=analysis_time,
            signal=result.get("signal"),
            conflict=result.get("conflict"),
            calls_made=result.get("calls_made", 1)
        )

        if response_cache and not request.deep_research:
            response_cache.set(cache_key, response)

        return response

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Documents =============

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index PDF/TXT/DOCX for RAG"""
    try:
        Validator.validate_file(file.filename, max_size=20*1024*1024)  # 20MB

        file_id      = str(uuid.uuid4())
        file_ext     = "." + file.filename.split(".")[-1]
        saved_name   = f"{file_id}{file_ext}"
        file_path    = os.path.join(UPLOAD_DIR, saved_name)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"File uploaded: {saved_name}")

        # Use new parent-child ingestion if retriever supports it
        doc_data = document_loader.load_and_chunk(file_path)
        chunks   = doc_data["chunks"]

        if hasattr(rag_retriever, "ingest_document"):
            # Tier 2 RAG — parent-child chunking
            full_text = " ".join(chunks)
            rag_retriever.ingest_document(
                text=full_text,
                doc_id=file_id,
                metadata={
                    "source":      file.filename,
                    "file_id":     file_id,
                    "upload_date": datetime.now().isoformat(),
                    "pages":       doc_data["metadata"].get("pages", "?")
                }
            )
        else:
            # Fallback: flat indexing
            doc_ids   = [f"{file_id}:{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": file.filename, "file_id": file_id,
                 "chunk_id": i, "upload_date": datetime.now().isoformat()}
                for i in range(len(chunks))
            ]
            rag_retriever.vector_store.add_documents(
                documents=chunks, ids=doc_ids, metadatas=metadatas
            )

        logger.info(f"Document indexed with {len(chunks)} chunks")

        return DocumentUploadResponse(
            file_id=file_id,
            filename=file.filename,
            status="indexed",
            chunks_created=len(chunks),
            upload_time=datetime.now().isoformat()
        )

    except DocumentException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get("/documents")
async def list_documents():
    try:
        files = os.listdir(UPLOAD_DIR)
        return ResponseFormatter.success(
            data={"files": files, "count": len(files)},
            message="Documents retrieved"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list documents")

@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    try:
        found = False
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(file_id):
                os.remove(os.path.join(UPLOAD_DIR, filename))
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Document not found")
        return ResponseFormatter.success(
            data={"file_id": file_id}, message="Document deleted"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete document")

# ============= Stock Analysis =============

@app.post("/analysis/stock", response_model=StockAnalysisResponse)
async def analyze_stock(request: StockAnalysisRequest):
    """
    Full stock analysis: technicals + fundamentals + regime detection.
    regime.current → trend / revert / volatile (HMM or KMeans on 90d data)
    """
    try:
        Validator.validate_ticker(request.ticker)
        ticker = request.ticker.upper()

        if rate_limiter and not rate_limiter.is_allowed(f"stock:{ticker}"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        logger.info(f"Analyzing stock: {ticker}")
        analysis = stock_analyzer.analyze_stock(ticker, request.analysis_type)

        if "error" in analysis:
            raise FinanceException(analysis["error"])

        # Sentiment
        sentiment_score = 0.0
        try:
            sent = sentiment_analyzer.analyze_ticker(ticker)
            sentiment_score = sent.get("overall_score", 0.0)
        except Exception as e:
            logger.warning(f"Sentiment failed: {e}")

        # Regime detection (runs in finance_module if regime_detector available)
        regime_data    = analysis.get("regime", {"current": "unknown"})
        regime_context = analysis.get("regime_context", "")

        # If finance_module didn't run it, try directly
        if regime_detector and regime_data.get("current") == "unknown":
            try:
                rd = regime_detector.detect(ticker)
                regime_data    = {
                    "current":  rd["current_regime"],
                    "probs":    rd.get("regime_probs", {}),
                    "dist_90d": rd.get("regime_dist_90d", {}),
                    "stats":    rd.get("stats", {}),
                    "method":   rd.get("method", "N/A")
                }
                regime_context = rd.get("llm_context", "")
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")

        return StockAnalysisResponse(
            ticker=ticker,
            price=analysis.get("price", 0.0),
            analysis=f"Technical Analysis for {ticker}",
            sentiment=sentiment_score,
            technical=analysis.get("technical", {}),
            fundamental=analysis.get("fundamental", {}),
            regime=regime_data,
            regime_context=regime_context,
            technicals_summary=analysis.get("technicals_summary", ""),
            timestamp=datetime.now().isoformat()
        )

    except FinanceException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Stock analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Stock analysis failed")

@app.get("/analysis/sentiment/{ticker}")
async def get_sentiment(ticker: str):
    try:
        Validator.validate_ticker(ticker)
        ticker = ticker.upper()
        sentiment = sentiment_analyzer.analyze_ticker(ticker)
        return ResponseFormatter.success(
            data=sentiment, message=f"Sentiment analysis for {ticker}"
        )
    except Exception as e:
        logger.error(f"Sentiment failed: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

@app.get("/analysis/watchlist")
async def get_watchlist_analysis():
    try:
        watchlist = ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]
        analyses  = []
        for ticker in watchlist:
            try:
                a = stock_analyzer.analyze_stock(ticker, analysis_type="technical")
                analyses.append({
                    "ticker":  ticker,
                    "price":   a.get("price", 0),
                    "signal":  a.get("technical", {}).get("signal", "N/A"),
                    "rsi":     a.get("technical", {}).get("rsi", 0),
                    "regime":  a.get("regime", {}).get("current", "unknown")
                })
            except Exception as e:
                logger.warning(f"Failed to analyze {ticker}: {e}")

        return ResponseFormatter.success(
            data={"analyses": analyses},
            message=f"Watchlist analysis ({len(analyses)}/{len(watchlist)})"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Watchlist analysis failed")

# ============= Backtest (placeholder) =============

@app.post("/backtest")
async def backtest_strategy(strategy_params: Dict):
    return ResponseFormatter.success(
        data={"status": "not_implemented"},
        message="Backtesting feature coming soon"
    )

# ============= Middleware & Error Handlers =============

@app.middleware("http")
async def log_requests(request, call_next):
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} - {response.status_code}")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

# ============= Entry Point =============

if __name__ == "__main__":
    import uvicorn
    if not check_initialization():
        logger.error("Initialization check failed")
        exit(1)
    uvicorn.run(app, host="0.0.0.0", port=API_PORT,
                reload=DEBUG, log_level="info")
