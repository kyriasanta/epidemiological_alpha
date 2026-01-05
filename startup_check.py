# check_environment.py - Run this when you start working
import sys
import subprocess

print("="*60)
print("ENVIRONMENT CHECK")
print("="*60)

# 1. Check Python
print(f"Python: {sys.version.split()[0]}")
print(f"Executable: {sys.executable}")
print(f"In venv: {'venv' in sys.executable}")

# 2. Check key packages
packages = ['pandas', 'numpy', 'yfinance', 'matplotlib']
print("\nKey packages:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} MISSING!")

# 3. Quick data test
print("\nQuick data test:")
try:
    import yfinance as yf
    import pandas as pd
    data = yf.download("AAPL", period="5d", progress=False)
    print(f"  ✓ yfinance working ({data.shape[0]} days of AAPL)")
except Exception as e:
    print(f"  ✗ Data test failed: {e}")

print("\n" + "="*60)
print("If all checks pass, you're ready to work!")
print("="*60)

# Optional: Auto-fix suggestions
if 'venv' not in sys.executable:
    print("\n⚠️  WARNING: Not in virtual environment!")
    print("Fix: In Terminal, run: venv\\Scripts\\activate")