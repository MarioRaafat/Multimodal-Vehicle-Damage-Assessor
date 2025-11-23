import os

################################### Paths ###################################
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
SEVERITY_DATA_DIR = os.path.join(DATA_DIR, 'Damage Severity Datasets', '1')
SEVERITY_TRAIN_DIR = os.path.join(SEVERITY_DATA_DIR, 'training')
SEVERITY_VAL_DIR = os.path.join(SEVERITY_DATA_DIR, 'validation')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')


################################### Parameters ###################################
# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'

################################### Models ###################################
SEVERITY_NUM_CLASSES = 3
SEVERITY_CLASS_NAMES = ['Minor', 'Moderate', 'Severe']
SEVERITY_CLASS_LABELS = {
    'minor': 0,
    'moderate': 1,
    'severe': 2
}

# Model architectures to compare (custom models)
SEVERITY_MODEL_ARCHITECTURES = [
    'simple_cnn',
    'vgg_style',
    'resnet_style',
    'efficientnet_b0',
    'mobilenet_v2'
]

# Pre-trained model architectures (RECOMMENDED - better performance)
PRETRAINED_MODEL_ARCHITECTURES = [
    'efficientnet_b3',    # Best overall efficiency
    'resnet50',           # Industry standard
    'densenet121',        # Efficient parameters
    'inceptionv3',        # Multi-scale features
    'xception',           # Depthwise separable convolutions
    'convnext_tiny',      # Modern CNN architecture
]

################################### Other Settings ###################################
# Random seed for reproducibility
RANDOM_SEED = 42


# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5