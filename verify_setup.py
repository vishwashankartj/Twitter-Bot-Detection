"""
Setup Verification Script
Checks that all dependencies and components are properly installed
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'networkx': 'networkx',
        'gensim': 'gensim',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'yaml': 'pyyaml',
        'kaggle': 'kaggle'
    }
    
    all_installed = True
    
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} (not installed)")
            all_installed = False
    
    return all_installed


def check_project_structure():
    """Check project directory structure"""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        'config',
        'data/raw',
        'data/processed',
        'data/external',
        'src/data',
        'src/models',
        'src/evaluation',
        'models',
        'results/plots',
        'notebooks'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def check_required_files():
    """Check required files"""
    print("\n📄 Checking required files...")
    
    required_files = [
        'requirements.txt',
        'config/config.yaml',
        'download_data.py',
        'train.py',
        'predict.py',
        'src/data/data_loader.py',
        'src/data/graph_builder.py',
        'src/data/feature_extractor.py',
        'src/models/graph2vec_embeddings.py',
        'src/models/classifiers.py',
        'src/evaluation/metrics.py',
        'README.md'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def check_kaggle_credentials():
    """Check Kaggle API credentials"""
    print("\n🔑 Checking Kaggle credentials...")
    
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json.exists():
        print(f"   ✅ Kaggle credentials found at {kaggle_json}")
        return True
    else:
        print(f"   ⚠️  Kaggle credentials not found (optional)")
        print(f"      To download datasets automatically, set up Kaggle API:")
        print(f"      https://www.kaggle.com/settings/account")
        return None  # Optional, not a failure


def main():
    """Run all verification checks"""
    print("=" * 70)
    print(" " * 15 + "TWITTER BOT DETECTION")
    print(" " * 20 + "Setup Verification")
    print("=" * 70)
    
    results = []
    
    # Run checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Required Files", check_required_files()))
    kaggle_result = check_kaggle_credentials()
    if kaggle_result is not None:
        results.append(("Kaggle Credentials", kaggle_result))
    
    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARY")
    print("=" * 70)
    
    all_passed = all(result for _, result in results)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Download dataset: python download_data.py")
        print("  2. Train model: python train.py")
        print("  3. Make predictions: python predict.py --input data.csv")
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
