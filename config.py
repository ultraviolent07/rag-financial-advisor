"""
Configuration management for the Trading Advisor
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent

# ============= API Configuration =============

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 1000
GROQ_TEMPERATURE = 0.7

# Optional: Claude fallback
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ============= Vector Store Configuration =============

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = "financial_docs"

# ============= Finance APIs =============

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ============= File Upload Configuration =============

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# ============= Server Configuration =============

API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ============= CORS Configuration =============

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

if not DEBUG:
    CORS_ORIGINS.extend([
        "https://yourdomain.com",
        "https://app.yourdomain.com",
    ])

# ============= Database Configuration =============

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./trading_advisor.db"
)

# ============= RAG Configuration =============

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Cosine similarity threshold

# ============= Sentiment Analysis =============

SENTIMENT_SOURCES = ["news", "options", "analyst", "social"]
SENTIMENT_MIN_CONFIDENCE = 0.5

# ============= Temporal RAG =============

# Weights for different time-based query types
TEMPORAL_WEIGHTS = {
    "current": {
        "very_recent": (0, 7),      # days_old range: (min, max)
        "weight": 1.0
    },
    "recent": {
        "range": (7, 30),
        "weight": 0.7
    },
    "historical": {
        "range": (30, 365),
        "weight": 0.3
    },
    "old": {
        "range": (365, 10000),
        "weight": 0.1
    }
}

# ============= Caching Configuration =============

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
CACHE_TTL = 3600  # Cache for 1 hour
CACHE_MAX_SIZE = 1000  # Max items in cache

# ============= Rate Limiting =============

RATE_LIMIT_ENABLED = True
RATE_LIMIT_CALLS_PER_MINUTE = 100
RATE_LIMIT_CALLS_PER_HOUR = 5000

# ============= Logging Configuration =============

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/trading_advisor.log")

# Create logs directory if it doesn't exist
log_dir = Path(LOG_FILE).parent
log_dir.mkdir(parents=True, exist_ok=True)

# ============= Feature Flags =============

ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_TEMPORAL_WEIGHTING = True
ENABLE_STREAMING = True
ENABLE_BACKTESTING = False  # Not yet implemented
ENABLE_KNOWLEDGE_GRAPH = False  # Not yet implemented

# ============= Data Sources Configuration =============

STOCK_DATA_SOURCE = "yfinance"  # yfinance or alpha_vantage
NEWS_DATA_SOURCE = "newsapi"  # newsapi or custom
SENTIMENT_DATA_SOURCES = ["finbert", "vader", "textblob"]

# ============= Validation =============

def validate_config():
    """Validate that all required config is set"""
    errors = []
    
    # Check Groq API key
    if not GROQ_API_KEY:
        errors.append("❌ GROQ_API_KEY not set in .env")
    
    # Check data directories exist
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  {error}")
        print("\nPlease set missing environment variables in .env file")
        return False
    
    return True

# ============= Configuration Summary =============

def print_config():
    """Print configuration summary (for debugging)"""
    print("\n=== Configuration Summary ===")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"API Port: {API_PORT}")
    print(f"Groq Model: {GROQ_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Chroma DB: {CHROMA_PERSIST_DIR}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    print(f"Features:")
    print(f"  - Sentiment Analysis: {ENABLE_SENTIMENT_ANALYSIS}")
    print(f"  - Temporal Weighting: {ENABLE_TEMPORAL_WEIGHTING}")
    print(f"  - Streaming: {ENABLE_STREAMING}")
    print(f"===========================\n")

if __name__ == "__main__":
    if validate_config():
        print("✅ Configuration valid!")
        print_config()
    else:
        print("❌ Configuration invalid!")
