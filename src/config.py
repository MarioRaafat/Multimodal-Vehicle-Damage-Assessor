import os
from pathlib import Path

# Project paths
# PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = Path('/content/gdrive/MyDrive/Multimodal-Vehicle-Damage-Assessor')
DATA_DIR = PROJECT_ROOT / "Data" / "Damage Severity Datasets" / "1"
TRAIN_DIR = DATA_DIR / "training"
VAL_DIR = DATA_DIR / "validation"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Class names and mapping
CLASS_NAMES = ['minor', 'moderate', 'severe']
CLASS_MAPPING = {
    '01-minor': 0,
    '02-moderate': 1,
    '03-severe': 2
}

# Image settings
IMG_SIZE = 224  # Standard size for most CNNs
YOLO_IMG_SIZE = 640  # Standard YOLO input size

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
YOLO_EPOCHS = 100

# Data augmentation settings
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# Model names
MODELS = {
    'yolov8': 'YOLOv8',
    'resnet50': 'ResNet50',
    'efficientnetb0': 'EfficientNetB0',
    'mobilenetv2': 'MobileNetV2'
}

# Device settings
DEVICE = 'cuda'  # Will be auto-detected in code (cuda/cpu)
