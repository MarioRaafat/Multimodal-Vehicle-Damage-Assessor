import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB4,
    ResNet50V2,
    MobileNetV2,
    DenseNet121,
    InceptionV3
)
from tensorflow.keras.regularizers import l2
import config


def create_classification_head(base_model, num_classes=3, dropout_rate=0.5):
    """
    Create a custom classification head for the pretrained model
    
    Args:
        base_model: Pretrained model (frozen or unfrozen)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Complete model with classification head
    """
    # Global Average Pooling to reduce dimensions
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(base_model.output)
    
    # Dense layer with regularization
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0001), name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Second dense layer
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0001), name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


def build_efficientnet_b4(input_shape, num_classes=3, weights='imagenet'):
    """Build EfficientNetB4 model"""
    base_model = EfficientNetB4(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = create_classification_head(base_model, num_classes)
    return model, base_model


def build_resnet50v2(input_shape, num_classes=3, weights='imagenet'):
    """Build ResNet50V2 model"""
    base_model = ResNet50V2(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = create_classification_head(base_model, num_classes)
    return model, base_model


def build_mobilenet_v2(input_shape, num_classes=3, weights='imagenet'):
    """Build MobileNetV2 model"""
    base_model = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        alpha=1.0  # Width multiplier
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = create_classification_head(base_model, num_classes)
    return model, base_model


def build_densenet121(input_shape, num_classes=3, weights='imagenet'):
    """Build DenseNet121 model"""
    base_model = DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = create_classification_head(base_model, num_classes)
    return model, base_model


def build_inceptionv3(input_shape, num_classes=3, weights='imagenet'):
    """Build InceptionV3 model"""
    base_model = InceptionV3(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = create_classification_head(base_model, num_classes)
    return model, base_model


def build_model(model_name, num_classes=3, weights='imagenet'):
    """
    Build a model by name
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        weights: Pretrained weights to use
        
    Returns:
        model: Complete model ready for training
        base_model: Base model for fine-tuning
    """
    # Get input shape for this model
    input_size = config.PRETRAINED_MODELS[model_name]['input_size']
    input_shape = (*input_size, config.IMG_CHANNELS)
    
    # Build model based on name
    model_builders = {
        'efficientnet_b4': build_efficientnet_b4,
        'resnet50v2': build_resnet50v2,
        'mobilenet_v2': build_mobilenet_v2,
        'densenet121': build_densenet121,
        'inceptionv3': build_inceptionv3
    }
    
    if model_name not in model_builders:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_builders.keys())}")
    
    print(f"\n{'='*60}")
    print(f"Building {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Description: {config.PRETRAINED_MODELS[model_name]['description']}")
    print(f"Parameters: {config.PRETRAINED_MODELS[model_name]['params']}")
    
    model, base_model = model_builders[model_name](input_shape, num_classes, weights)
    
    print(f"Total layers: {len(model.layers)}")
    print(f"Trainable params: {model.count_params():,}")
    print(f"{'='*60}\n")
    
    return model, base_model


def unfreeze_model(model, base_model, num_layers_to_unfreeze):
    """
    Unfreeze the top layers of the base model for fine-tuning
    
    Args:
        model: Complete model
        base_model: Base pretrained model
        num_layers_to_unfreeze: Number of layers from top to unfreeze
    """
    base_model.trainable = True
    
    # Freeze all layers except the last num_layers_to_unfreeze
    total_layers = len(base_model.layers)
    freeze_until = total_layers - num_layers_to_unfreeze
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"\n{'='*60}")
    print(f"FINE-TUNING MODE")
    print(f"{'='*60}")
    print(f"Total layers in base model: {total_layers}")
    print(f"Unfrozen layers: {num_layers_to_unfreeze}")
    print(f"Total trainable layers: {trainable_count}")
    print(f"{'='*60}\n")
    
    return model


def get_model_summary(model):
    """Get a formatted summary of the model"""
    import io
    import sys
    
    # Capture model.summary() output
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    summary_string = stream.getvalue()
    
    return summary_string
