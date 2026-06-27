"""
Setup verification script
Checks all dependencies and configurations before running
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("\nChecking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"ERROR: Python 3.9+ required, found {version.major}.{version.minor}")
        return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    print("\nChecking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual environment active at: {sys.prefix}")
        return True
    else:
        print("WARNING: Not running in virtual environment")
        print("Consider activating: source venv/bin/activate")
        return True

def check_required_packages():
    """Check required packages are installed"""
    print("\nChecking required packages...")
    required = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'chromadb',
        'sentence_transformers',
        'transformers',
        'groq',
        'yfinance',
        'pdfplumber',
        'docx',
        'requests',
        'dotenv'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  {package} - OK")
        except ImportError:
            print(f"  {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nERROR: Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_environment_file():
    """Check .env file exists and has required variables"""
    print("\nChecking environment file...")
    env_path = Path(".env")
    
    if not env_path.exists():
        print("ERROR: .env file not found")
        print("Create .env with:")
        print("  GROQ_API_KEY=gsk_your_key")
        print("  NEWS_API_KEY=your_key (optional)")
        print("  API_PORT=8000")
        return False
    
    # Load and check
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set in .env")
        return False
    
    print(f"  GROQ_API_KEY: {groq_key[:20]}... - OK")
    print(f"  API_PORT: {os.getenv('API_PORT', '8000')} - OK")
    
    return True

def check_directories():
    """Check required directories exist"""
    print("\nChecking directories...")
    directories = [
        "data",
        "data/chroma_db",
        "data/uploads"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {directory} - OK")
    
    return True

def test_groq_connection():
    """Test Groq API connection"""
    print("\nTesting Groq API connection...")
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  Skipped (no API key)")
        return True
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=50
        )
        
        if response.choices[0].message.content:
            print("  Groq API connection - OK")
            return True
        else:
            print("  ERROR: Groq API no response")
            return False
            
    except Exception as e:
        print(f"  ERROR: Groq API test failed - {e}")
        return False

def test_embeddings():
    """Test sentence transformers"""
    print("\nTesting embeddings model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(["test sentence"])
        print(f"  Embeddings model loaded - OK")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        return True
    except Exception as e:
        print(f"  ERROR: Embeddings test failed - {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline initialization"""
    print("\nTesting RAG pipeline...")
    try:
        from rag_pipeline import initialize_rag_pipeline
        retriever = initialize_rag_pipeline()
        print("  RAG pipeline initialized - OK")
        return True
    except Exception as e:
        print(f"  ERROR: RAG pipeline test failed - {e}")
        return False

def test_finance_module():
    """Test finance module"""
    print("\nTesting finance module...")
    try:
        from finance_module import StockAnalysisService
        service = StockAnalysisService()
        print("  Finance service initialized - OK")
        return True
    except Exception as e:
        print(f"  ERROR: Finance module test failed - {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 50)
    print("Trading Advisor - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Required Packages", check_required_packages),
        ("Environment File", check_environment_file),
        ("Directories", check_directories),
        ("Embeddings Model", test_embeddings),
        ("RAG Pipeline", test_rag_pipeline),
        ("Finance Module", test_finance_module),
        ("Groq Connection", test_groq_connection),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\nAll checks passed! Ready to run:")
        print("  uvicorn backend_main_final:app --reload")
    else:
        print("\nSome checks failed. Fix issues above before running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
