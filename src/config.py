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
EPOCHS = 150
INITIAL_LEARNING_RATE = 0.0001  # Lower for fine-tuning pretrained models
VALIDATION_SPLIT = 0.2

# Fine-tuning parameters
UNFREEZE_LAYERS = 50  # Number of layers to unfreeze from the top
FINE_TUNE_EPOCHS = 30  # Additional epochs for fine-tuning
FINE_TUNE_LR = 0.00001  # Lower learning rate for fine-tuning

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
PRETRAINED_MODELS = {
    'efficientnet_b4': {
        'input_size': (380, 380),
        'description': 'Best accuracy/efficiency tradeoff',
        'params': '19M'
    },
    'resnet50v2': {
        'input_size': (224, 224),
        'description': 'Industry standard, reliable',
        'params': '25M'
    },
    'mobilenet_v2': {
        'input_size': (224, 224),
        'description': 'Lightweight, fast inference',
        'params': '3.5M'
    },
    'densenet121': {
        'input_size': (224, 224),
        'description': 'Efficient feature reuse',
        'params': '8M'
    },
    'inceptionv3': {
        'input_size': (299, 299),
        'description': 'Multi-scale feature extraction',
        'params': '23M'
    }
}

# Default models to train (can be modified)
MODELS_TO_TRAIN = ['efficientnet_b4', 'resnet50v2', 'densenet121', 'mobilenet_v2', 'inceptionv3']

################################### Other Settings ###################################
# Random seed for reproducibility
RANDOM_SEED = 42


# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5