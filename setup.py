#!/usr/bin/env python3
"""
Setup script for Trading Advisor
Initializes the project, checks dependencies, and configures environment
Run this first: python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple

PYTHON_MIN_VERSION = (3, 9)
PROJECT_ROOT = Path(__file__).parent

# ============= Color Output =============

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {msg}")

# ============= Checks =============

def check_python_version() -> bool:
    """Check if Python version is acceptable"""
    print_info("Checking Python version...")
    
    version = sys.version_info
    if version < PYTHON_MIN_VERSION:
        print_error(f"Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+ required, got {version.major}.{version.minor}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_project_structure() -> bool:
    """Check if all required files exist"""
    print_info("Checking project structure...")
    
    required_files = [
        "config.py",
        "utils.py",
        "pdf_parser.py",
        "rag_pipeline.py",
        "groq_integration.py",
        "finance_module.py",
        "multi_source_sentiment.py",
        "backend_main_complete.py",
        "requirements.txt",
        "RUN_GUIDE.md"
    ]
    
    missing = []
    for file in required_files:
        if not (PROJECT_ROOT / file).exists():
            missing.append(file)
    
    if missing:
        print_warning(f"Missing files: {', '.join(missing)}")
        print_info("Some files may not be critical for startup")
    else:
        print_success("All required files present")
    
    return True

# ============= Directory Setup =============

def create_directories() -> bool:
    """Create necessary directories"""
    print_info("Creating directories...")
    
    directories = [
        "data",
        "data/uploads",
        "data/chroma_db",
        "logs"
    ]
    
    for dir_path in directories:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created/verified: {dir_path}")
    
    return True

# ============= Environment Setup =============

def create_env_file() -> bool:
    """Create .env file if it doesn't exist"""
    print_info("Setting up .env file...")
    
    env_file = PROJECT_ROOT / ".env"
    
    if env_file.exists():
        print_warning(".env already exists, skipping creation")
        return True
    
    env_template = """# Groq LLM Configuration (REQUIRED)
GROQ_API_KEY=gsk_YOUR_KEY_HERE

# Optional: Finance APIs
NEWS_API_KEY=your_newsapi_key
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Data Directories
CHROMA_PERSIST_DIR=./data/chroma_db
UPLOAD_DIR=./data/uploads

# Server Configuration
API_PORT=8000
DEBUG=True
ENVIRONMENT=development
LOG_LEVEL=INFO
LOG_FILE=./logs/trading_advisor.log

# Features
ENABLE_SENTIMENT_ANALYSIS=True
ENABLE_TEMPORAL_WEIGHTING=True
ENABLE_STREAMING=True
CACHE_ENABLED=True
RATE_LIMIT_ENABLED=True
"""
    
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print_success(".env file created")
    print_warning("IMPORTANT: Update GROQ_API_KEY in .env file")
    print_info("Get free key from: https://console.groq.com/keys")
    
    return True

# ============= Virtual Environment =============

def check_venv() -> Tuple[bool, str]:
    """Check if virtual environment exists and is activated"""
    print_info("Checking virtual environment...")
    
    venv_path = PROJECT_ROOT / "venv"
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not venv_path.exists():
        print_warning("Virtual environment not found")
        print_info("Create with: python -m venv venv")
        print_info("Activate with: source venv/bin/activate  (or venv\\Scripts\\activate on Windows)")
        return False, "venv not created"
    
    if not in_venv:
        print_warning("Virtual environment not activated")
        print_info("Activate with: source venv/bin/activate")
        return False, "venv not activated"
    
    print_success("Virtual environment ready")
    return True, "venv ready"

# ============= Dependencies =============

def check_dependencies() -> bool:
    """Check if all dependencies are installed"""
    print_info("Checking dependencies...")
    
    critical_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "chromadb",
        "sentence_transformers",
        "groq",
        "yfinance",
        "pdfplumber",
        "transformers"
    ]
    
    missing = []
    for package in critical_packages:
        try:
            __import__(package)
            print_success(f"Found: {package}")
        except ImportError:
            print_warning(f"Missing: {package}")
            missing.append(package)
    
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        print_info("Install with: pip install -r requirements.txt")
        return False
    
    print_success("All critical dependencies installed")
    return True

# ============= Configuration Validation =============

def validate_configuration() -> bool:
    """Validate configuration"""
    print_info("Validating configuration...")
    
    env_file = PROJECT_ROOT / ".env"
    
    if not env_file.exists():
        print_error(".env file not found")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if not groq_key or groq_key == "gsk_YOUR_KEY_HERE":
        print_error("GROQ_API_KEY not set or invalid in .env")
        print_info("Update .env with your key from https://console.groq.com/keys")
        return False
    
    print_success("GROQ_API_KEY configured")
    
    # Try to import config
    try:
        from config import validate_config
        if validate_config():
            print_success("Configuration validation passed")
            return True
        else:
            print_error("Configuration validation failed")
            return False
    except Exception as e:
        print_error(f"Configuration error: {e}")
        return False

# ============= Groq Test =============

def test_groq() -> bool:
    """Test Groq API connection"""
    print_info("Testing Groq API connection...")
    
    try:
        from groq import Groq
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            print_error("GROQ_API_KEY not set")
            return False
        
        client = Groq(api_key=api_key)
        
        print_info("Sending test request to Groq...")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'Ready'"}],
            model="llama-3.3-70b-versatile",
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print_success(f"Groq API working: '{result}'")
        return True
        
    except Exception as e:
        print_error(f"Groq API test failed: {e}")
        return False

# ============= RAG Test =============

def test_rag() -> bool:
    """Test RAG pipeline"""
    print_info("Testing RAG pipeline...")
    
    try:
        from rag_pipeline import initialize_rag_pipeline
        
        retriever = initialize_rag_pipeline()
        print_success("RAG pipeline initialized")
        
        # Test with sample docs
        sample_docs = [
            "Apple reported Q3 earnings",
            "Microsoft Azure growth"
        ]
        retriever.vector_store.add_documents(
            documents=sample_docs,
            ids=["test_1", "test_2"]
        )
        
        print_success("RAG pipeline working")
        return True
        
    except Exception as e:
        print_error(f"RAG pipeline test failed: {e}")
        return False

# ============= Main Setup Routine =============

def main():
    """Run setup"""
    print(f"\n{Colors.BLUE}=== Trading Advisor Setup ==={Colors.RESET}\n")
    
    steps = [
        ("Python version", check_python_version),
        ("Project structure", check_project_structure),
        ("Directories", create_directories),
        (".env file", create_env_file),
        ("Configuration", validate_configuration),
        ("Dependencies", check_dependencies),
    ]
    
    passed = 0
    failed = 0
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_error(f"{step_name} failed: {e}")
            failed += 1
    
    print(f"\n{Colors.BLUE}=== Setup Results ==={Colors.RESET}")
    print_success(f"{passed} checks passed")
    
    if failed > 0:
        print_error(f"{failed} checks failed")
        print_info("Fix errors above before proceeding")
        return False
    
    # Optional tests
    print(f"\n{Colors.BLUE}=== Optional Tests ==={Colors.RESET}\n")
    
    try:
        venv_ok, msg = check_venv()
        if venv_ok:
            test_groq()
            test_rag()
    except Exception as e:
        print_warning(f"Optional tests skipped: {e}")
    
    # Next steps
    print(f"\n{Colors.BLUE}=== Next Steps ==={Colors.RESET}\n")
    print_info("1. Update GROQ_API_KEY in .env file")
    print_info("2. Run: python backend_main_complete.py")
    print_info("3. In another terminal, run frontend: cd frontend && npm run dev")
    print_info("4. Open: http://localhost:3000")
    
    print(f"\n{Colors.GREEN}Setup complete!{Colors.RESET}\n")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
