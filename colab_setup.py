import os
import sys

def setup_colab_environment():
    try:
        import google.colab
        print("✓ Running in Google Colab")
    except ImportError:
        print("⚠ Not running in Google Colab, but continuing setup...")
    
    # Install required packages
    print("\n📦 Installing required packages...")
    os.system("pip install -q tensorflow matplotlib seaborn pillow scikit-learn")
    
    # Set up project paths
    project_root = '/content/Multimodal-Vehicle-Damage-Assessor'
    
    # Add src to path if not already there
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"✓ Added {src_path} to Python path")
    
    # Create necessary directories
    dirs_to_create = [
        os.path.join(project_root, 'results'),
        os.path.join(project_root, 'logs'),
        os.path.join(project_root, 'models')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ Created necessary directories")
    
    # Change to project root
    os.chdir(project_root)
    print(f"✓ Changed working directory to {project_root}")
    
    print("\n✅ Setup complete! You can now run the notebooks.")
    print("\n📋 Next steps:")
    print("1. Upload your Data folder to /content/Multimodal-Vehicle-Damage-Assessor/")
    print("2. Run the notebooks in order: 01_data_exploration.ipynb → 02_train_all_models.ipynb → 03_model_comparison.ipynb")
    
    return project_root

if __name__ == "__main__":
    setup_colab_environment()
