"""
Error handling and utility functions
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import traceback

from config import LOG_LEVEL, LOG_FILE

# ============= Setup Logging =============

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

def setup_logger(name: str) -> logging.Logger:
    """Setup logger for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger(__name__)

# ============= Custom Exceptions =============

class TradingAdvisorException(Exception):
    """Base exception for Trading Advisor"""
    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)


class ConfigurationException(TradingAdvisorException):
    """Raised when configuration is invalid"""
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")


class RAGException(TradingAdvisorException):
    """Raised when RAG pipeline fails"""
    def __init__(self, message: str):
        super().__init__(message, "RAG_ERROR")


class LLMException(TradingAdvisorException):
    """Raised when LLM call fails"""
    def __init__(self, message: str):
        super().__init__(message, "LLM_ERROR")


class FinanceException(TradingAdvisorException):
    """Raised when finance data fetch fails"""
    def __init__(self, message: str):
        super().__init__(message, "FINANCE_ERROR")


class DocumentException(TradingAdvisorException):
    """Raised when document processing fails"""
    def __init__(self, message: str):
        super().__init__(message, "DOCUMENT_ERROR")


class RateLimitException(TradingAdvisorException):
    """Raised when rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT_ERROR")


# ============= Error Response =============

class ErrorResponse:
    """Standardized error response format"""
    
    @staticmethod
    def format(
        error: Exception,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format error as JSON response
        """
        error_message = str(error)
        error_code = "UNKNOWN_ERROR"
        
        if isinstance(error, TradingAdvisorException):
            error_code = error.error_code
            error_message = error.message
        
        response = {
            "error": True,
            "status_code": status_code,
            "error_code": error_code,
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
        }
        
        if details:
            response["details"] = details
        
        # Log error
        logger.error(f"[{error_code}] {error_message}")
        if hasattr(error, '__traceback__'):
            logger.error(traceback.format_exc())
        
        return response


# ============= Validation Utilities =============

class Validator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_query(query: str, min_length: int = 3, max_length: int = 1000) -> bool:
        """Validate user query"""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        query = query.strip()
        
        if len(query) < min_length:
            raise ValueError(f"Query must be at least {min_length} characters")
        
        if len(query) > max_length:
            raise ValueError(f"Query must be at most {max_length} characters")
        
        return True
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate stock ticker"""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        ticker = ticker.strip().upper()
        
        if len(ticker) < 1 or len(ticker) > 5:
            raise ValueError("Ticker must be 1-5 characters")
        
        if not ticker.isalpha():
            raise ValueError("Ticker must contain only letters")
        
        return True
    
    @staticmethod
    def validate_conversation_history(history: list) -> bool:
        """Validate conversation history format"""
        if not isinstance(history, list):
            raise ValueError("Conversation history must be a list")
        
        for item in history:
            if not isinstance(item, dict):
                raise ValueError("Each message must be a dictionary")
            
            if "role" not in item or "content" not in item:
                raise ValueError("Each message must have 'role' and 'content'")
            
            if item["role"] not in ["user", "assistant"]:
                raise ValueError("Role must be 'user' or 'assistant'")
        
        return True
    
    @staticmethod
    def validate_file(filename: str, max_size: int) -> bool:
        """Validate uploaded file"""
        from config import ALLOWED_EXTENSIONS
        
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        
        if file_ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}")
        
        return True


# ============= Rate Limiting =============

class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Args:
            max_calls: Max number of calls
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        from time import time
        
        current_time = time()
        
        if identifier not in self.calls:
            self.calls[identifier] = []
        
        # Remove old calls outside time window
        self.calls[identifier] = [
            call_time for call_time in self.calls[identifier]
            if current_time - call_time < self.time_window
        ]
        
        if len(self.calls[identifier]) >= self.max_calls:
            return False
        
        self.calls[identifier].append(current_time)
        return True
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining calls for identifier"""
        from time import time
        
        current_time = time()
        
        if identifier not in self.calls:
            return self.max_calls
        
        # Remove old calls
        self.calls[identifier] = [
            call_time for call_time in self.calls[identifier]
            if current_time - call_time < self.time_window
        ]
        
        return max(0, self.max_calls - len(self.calls[identifier]))


# ============= Cache Utilities =============

class SimpleCache:
    """Simple in-memory cache"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """
        Args:
            ttl: Time to live in seconds
            max_size: Maximum cache size
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        from time import time
        
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        if time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        from time import time
        
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time())
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


# ============= Response Formatting =============

class ResponseFormatter:
    """Format responses consistently"""
    
    @staticmethod
    def success(
        data: Any,
        message: str = "Success",
        status_code: int = 200
    ) -> Dict[str, Any]:
        """Format success response"""
        return {
            "success": True,
            "status_code": status_code,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def error(
        error: Exception,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format error response"""
        return ErrorResponse.format(error, status_code, details)
    
    @staticmethod
    def list_response(
        items: list,
        total: int,
        page: int = 1,
        per_page: int = 10
    ) -> Dict[str, Any]:
        """Format list response with pagination"""
        return {
            "success": True,
            "items": items,
            "pagination": {
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": (total + per_page - 1) // per_page
            },
            "timestamp": datetime.now().isoformat()
        }


# ============= Helper Functions =============

def log_request(method: str, path: str, status_code: int) -> None:
    """Log incoming request"""
    logger.info(f"{method} {path} - {status_code}")

def log_error(error: Exception, context: str = "") -> None:
    """Log error with context"""
    logger.error(f"Error in {context}: {str(error)}")
    logger.error(traceback.format_exc())

def safe_execute(func, *args, **kwargs) -> tuple:
    """
    Safely execute function and return (success, result/error)
    
    Returns:
        (success: bool, result: Any)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        log_error(e, func.__name__)
        return False, e

# ============= Initialization Check =============

def check_initialization() -> bool:
    """Check if all required components are initialized"""
    try:
        # Check GROQ API key
        from config import GROQ_API_KEY, validate_config
        
        if not validate_config():
            logger.error("Configuration validation failed")
            return False
        
        logger.info("All initialization checks passed")
        return True
        
    except Exception as e:
        logger.error(f"Initialization check failed: {e}")
        return False
