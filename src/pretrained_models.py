import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152,
    InceptionV3, InceptionResNetV2,
    DenseNet121, DenseNet169, DenseNet201,
    EfficientNetB0, EfficientNetB3, EfficientNetB7,
    MobileNetV2, MobileNetV3Large,
    VGG16, VGG19,
    Xception,
    NASNetLarge,
    ConvNeXtTiny, ConvNeXtBase
)
import config


def create_transfer_learning_model(base_model, model_name, trainable_layers=0):
    """
    Generic function to create transfer learning model from any base model.
    
    Args:
        base_model: Pre-trained Keras model
        model_name: Name for the model
        trainable_layers: Number of layers from the end to make trainable (0 = freeze all)
    """
    # Freeze the base model initially
    base_model.trainable = False
    
    # If trainable_layers > 0, unfreeze last N layers
    if trainable_layers > 0:
        base_model.trainable = True
        # Freeze all layers except the last trainable_layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    
    # Build the model
    inputs = layers.Input(shape=config.INPUT_SHAPE)
    
    # Preprocessing for the specific model (if needed)
    x = inputs
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(config.SEVERITY_NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


# ResNet Family - Deep residual networks
def create_resnet50():
    """ResNet50 - 50 layers, excellent balance of speed and accuracy"""
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'resnet50')


def create_resnet101():
    """ResNet101 - 101 layers, more capacity than ResNet50"""
    base_model = ResNet101(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'resnet101')


def create_resnet152():
    """ResNet152 - 152 layers, highest capacity in ResNet family"""
    base_model = ResNet152(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'resnet152')


# Inception Family - Multi-scale feature extraction
def create_inceptionv3():
    """InceptionV3 - Efficient multi-scale convolutions"""
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'inceptionv3')


def create_inception_resnet_v2():
    """InceptionResNetV2 - Combines Inception and ResNet"""
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'inception_resnet_v2')


# DenseNet Family - Dense connections between layers
def create_densenet121():
    """DenseNet121 - Efficient parameter usage with dense connections"""
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'densenet121')


def create_densenet169():
    """DenseNet169 - Deeper than DenseNet121"""
    base_model = DenseNet169(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'densenet169')


def create_densenet201():
    """DenseNet201 - Deepest in DenseNet family"""
    base_model = DenseNet201(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'densenet201')


# EfficientNet Family - State-of-the-art efficiency
def create_efficientnet_b0():
    """EfficientNetB0 - Smallest, fastest EfficientNet"""
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'efficientnet_b0')


def create_efficientnet_b3():
    """EfficientNetB3 - Balanced performance"""
    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'efficientnet_b3')


def create_efficientnet_b7():
    """EfficientNetB7 - Highest accuracy in EfficientNet family"""
    base_model = EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'efficientnet_b7')


# MobileNet Family - Optimized for mobile/edge devices
def create_mobilenet_v2():
    """MobileNetV2 - Fast and efficient"""
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'mobilenet_v2')


def create_mobilenet_v3_large():
    """MobileNetV3Large - Improved MobileNet architecture"""
    base_model = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE,
        minimalistic=False
    )
    return create_transfer_learning_model(base_model, 'mobilenet_v3_large')


# VGG Family - Classic architecture
def create_vgg16():
    """VGG16 - Classic deep CNN architecture"""
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'vgg16')


def create_vgg19():
    """VGG19 - Deeper VGG architecture"""
    base_model = VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'vgg19')


# Xception - Extreme Inception
def create_xception():
    """Xception - Depthwise separable convolutions"""
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'xception')


# ConvNeXt Family - Modern CNN architecture (2022)
def create_convnext_tiny():
    """ConvNeXtTiny - Modern pure CNN architecture"""
    base_model = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'convnext_tiny')


def create_convnext_base():
    """ConvNeXtBase - Larger ConvNeXt model"""
    base_model = ConvNeXtBase(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE
    )
    return create_transfer_learning_model(base_model, 'convnext_base')


def get_pretrained_model(architecture_name):
    model_dict = {
        # ResNet Family
        'resnet50': create_resnet50,
        'resnet101': create_resnet101,
        'resnet152': create_resnet152,
        
        # Inception Family
        'inceptionv3': create_inceptionv3,
        'inception_resnet_v2': create_inception_resnet_v2,
        
        # DenseNet Family
        'densenet121': create_densenet121,
        'densenet169': create_densenet169,
        'densenet201': create_densenet201,
        
        # EfficientNet Family
        'efficientnet_b0': create_efficientnet_b0,
        'efficientnet_b3': create_efficientnet_b3,
        'efficientnet_b7': create_efficientnet_b7,
        
        # MobileNet Family
        'mobilenet_v2': create_mobilenet_v2,
        'mobilenet_v3_large': create_mobilenet_v3_large,
        
        # VGG Family
        'vgg16': create_vgg16,
        'vgg19': create_vgg19,
        
        # Others
        'xception': create_xception,
        'convnext_tiny': create_convnext_tiny,
        'convnext_base': create_convnext_base,
    }
    
    if architecture_name not in model_dict:
        available = ', '.join(model_dict.keys())
        raise ValueError(f"Unknown architecture: {architecture_name}. Available: {available}")
    
    return model_dict[architecture_name]()


def compile_pretrained_model(model, learning_rate=None):
    """Compile model with appropriate settings for transfer learning"""
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


# Model recommendations by use case
RECOMMENDED_MODELS = {
    'fast': ['efficientnet_b0', 'mobilenet_v2', 'mobilenet_v3_large'],
    'balanced': ['resnet50', 'densenet121', 'inceptionv3', 'convnext_tiny'],
    'accurate': ['efficientnet_b7', 'resnet152', 'inception_resnet_v2', 'densenet201'],
    'all_top_performers': [
        'efficientnet_b3',  # Best overall efficiency
        'resnet50',          # Industry standard
        'densenet121',       # Efficient parameters
        'inceptionv3',       # Multi-scale features
        'xception',          # Depthwise separable convs
        'convnext_tiny',     # Modern architecture
    ]
}


def get_recommended_models(category='all_top_performers'):
    return RECOMMENDED_MODELS.get(category, RECOMMENDED_MODELS['all_top_performers'])
