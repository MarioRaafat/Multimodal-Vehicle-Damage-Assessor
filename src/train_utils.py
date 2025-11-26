import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, MobileNetV2, VGG16
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

from config import (
    IMG_SIZE, LEARNING_RATE, CLASS_NAMES, 
    MODELS_DIR, LOGS_DIR, RANDOM_SEED
)


def build_transfer_learning_model(base_model_name, img_size=IMG_SIZE, num_classes=3):
    """
    Build a transfer learning model with a specified base
    
    Args:
        base_model_name: Name of the base model ('resnet50', 'efficientnetb0', 'mobilenetv2', 'vgg16')
        img_size: Input image size
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Select base model
    base_models = {
        'resnet50': ResNet50,
        'efficientnetb0': EfficientNetB0,
        'mobilenetv2': MobileNetV2,
        'vgg16': VGG16
    }
    
    if base_model_name.lower() not in base_models:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Load base model
    base_model = base_models[base_model_name.lower()](
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_name, patience=10):
    """
    Get training callbacks
    
    Args:
        model_name: Name of the model for saving
        patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(LOGS_DIR / model_name),
            histogram_freq=1
        )
    ]
    
    return callbacks


def evaluate_model(model, data_generator):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained Keras model
        data_generator: Data generator
        
    Returns:
        Dictionary with metrics
    """
    # Get predictions
    predictions = model.predict(data_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = data_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class_metrics': {
            CLASS_NAMES[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
            for i in range(len(CLASS_NAMES))
        },
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist()
    }
    
    return results


def save_model_results(model_name, results, history=None):
    """
    Save model results to JSON file
    
    Args:
        model_name: Name of the model
        results: Results dictionary from evaluate_model
        history: Training history (optional)
    """
    output = {
        'model_name': model_name,
        'metrics': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        },
        'per_class_metrics': results['per_class_metrics']
    }
    
    if history is not None:
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
            
        output['training_history'] = {
            'final_train_accuracy': float(history_dict['accuracy'][-1]),
            'final_val_accuracy': float(history_dict['val_accuracy'][-1]),
            'final_train_loss': float(history_dict['loss'][-1]),
            'final_val_loss': float(history_dict['val_loss'][-1]),
            'epochs_trained': len(history_dict['accuracy'])
        }
    
    # Save to file
    save_path = MODELS_DIR / f'{model_name}_results.json'
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Saved results to {save_path}")


def fine_tune_model(model, train_generator, val_generator, 
                    base_layers_to_unfreeze=20, epochs=20):
    """
    Fine-tune a pre-trained model by unfreezing some layers
    
    Args:
        model: Pre-trained model
        train_generator: Training data generator
        val_generator: Validation data generator
        base_layers_to_unfreeze: Number of base layers to unfreeze
        epochs: Number of fine-tuning epochs
        
    Returns:
        Training history
    """
    # Unfreeze the base model layers
    base_model = model.layers[1]  # Assuming the base model is the second layer
    base_model.trainable = True
    
    # Freeze all layers except the last N
    for layer in base_model.layers[:-base_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Fine-tuning last {base_layers_to_unfreeze} layers...")
    
    # Train
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1
    )
    
    return history


def load_trained_model(model_name):
    """
    Load a trained model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Loaded Keras model
    """
    model_path = MODELS_DIR / f'{model_name}_best.h5'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    return model
