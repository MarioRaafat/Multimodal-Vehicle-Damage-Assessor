"""
Quick test to verify all dependencies and project setup
"""
import sys
import os

print("="*80)
print("DEPENDENCY CHECK - Multimodal Vehicle Damage Assessor")
print("="*80)

errors = []
warnings = []

# Test 1: Python version
print("\n[1/8] Checking Python version...")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 8:
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    errors.append(f"Python 3.8+ required (found {python_version.major}.{python_version.minor})")
    print(f"✗ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

# Test 2: TensorFlow
print("\n[2/8] Checking TensorFlow...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ GPU available: {len(gpus)} device(s)")
    else:
        warnings.append("No GPU detected - training will be slower")
        print("  ⚠ No GPU detected (CPU will be used)")
except ImportError as e:
    errors.append("TensorFlow not installed")
    print(f"✗ TensorFlow not found")

# Test 3: NumPy
print("\n[3/8] Checking NumPy...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    errors.append("NumPy not installed")
    print("✗ NumPy not found")

# Test 4: Matplotlib
print("\n[4/8] Checking Matplotlib...")
try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    errors.append("Matplotlib not installed")
    print("✗ Matplotlib not found")

# Test 5: Seaborn
print("\n[5/8] Checking Seaborn...")
try:
    import seaborn as sns
    print(f"✓ Seaborn {sns.__version__}")
except ImportError:
    errors.append("Seaborn not installed")
    print("✗ Seaborn not found")

# Test 6: Scikit-learn
print("\n[6/8] Checking Scikit-learn...")
try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError:
    errors.append("Scikit-learn not installed")
    print("✗ Scikit-learn not found")

# Test 7: Pandas
print("\n[7/8] Checking Pandas...")
try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError:
    errors.append("Pandas not installed")
    print("✗ Pandas not found")

# Test 8: PIL/Pillow
print("\n[8/8] Checking Pillow...")
try:
    from PIL import Image
    import PIL
    print(f"✓ Pillow {PIL.__version__}")
except ImportError:
    errors.append("Pillow not installed")
    print("✗ Pillow not found")

# Check project structure
print("\n" + "="*80)
print("PROJECT STRUCTURE CHECK")
print("="*80)

# Add src to path for config import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("\n[1/3] Checking source files...")
required_files = [
    'src/config.py',
    'src/data_loader.py',
    'src/models.py',
    'src/train.py',
    'src/visualize.py'
]

for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        errors.append(f"Missing file: {file}")
        print(f"✗ {file}")

print("\n[2/3] Checking dataset...")
try:
    import config
    
    if os.path.exists(config.SEVERITY_TRAIN_DIR):
        print(f"✓ Training directory: {config.SEVERITY_TRAIN_DIR}")
        
        # Count images
        total_train = 0
        for class_folder in ['01-minor', '02-moderate', '03-severe']:
            class_dir = os.path.join(config.SEVERITY_TRAIN_DIR, class_folder)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {class_folder}: {count} images")
                total_train += count
        print(f"  Total training images: {total_train}")
        
        if total_train < 100:
            warnings.append("Low number of training images - consider adding more data")
    else:
        errors.append(f"Training directory not found: {config.SEVERITY_TRAIN_DIR}")
        print(f"✗ Training directory not found")
    
    if os.path.exists(config.SEVERITY_VAL_DIR):
        print(f"✓ Validation directory: {config.SEVERITY_VAL_DIR}")
        
        # Count images
        total_val = 0
        for class_folder in ['01-minor', '02-moderate', '03-severe']:
            class_dir = os.path.join(config.SEVERITY_VAL_DIR, class_folder)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_val += count
        print(f"  Total validation images: {total_val}")
        
        if total_val < 30:
            warnings.append("Low number of validation images - consider adding more data")
    else:
        errors.append(f"Validation directory not found: {config.SEVERITY_VAL_DIR}")
        print(f"✗ Validation directory not found")
        
except ImportError:
    errors.append("Cannot import config.py")
    print("✗ Cannot import config.py")

print("\n[3/3] Checking output directories...")
output_dirs = ['models', 'results', 'logs']
for dir_name in output_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created {dir_name}/")
    else:
        print(f"✓ {dir_name}/")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if errors:
    print("\n❌ ERRORS FOUND:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\nPlease fix these errors before proceeding.")
    print("Run: pip install -r requirements.txt")
else:
    print("\n✅ All required dependencies are installed!")

if warnings:
    print("\n⚠️  WARNINGS:")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")

if not errors:
    print("\n" + "="*80)
    print("🚀 READY TO START!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run setup: python setup.py")
    print("  2. Explore data: python notebooks/01_data_exploration.py")
    print("  3. Train models: python notebooks/02_train_all_models.py")
    print("  4. Compare results: python notebooks/03_model_comparison.py")
    print("\nOr run everything: python run_all.py")
    print("="*80)
else:
    print("\n" + "="*80)
    print("❌ SETUP INCOMPLETE")
    print("="*80)
    print("\nPlease address the errors above before training.")

print()
