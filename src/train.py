import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
import config


def get_callbacks(model_name, log_dir=None):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(config.MODELS_DIR, f'{model_name}_best.keras')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    )
    
    # CSV logger
    csv_path = os.path.join(config.LOGS_DIR, f'{model_name}_training.csv')
    csv_logger = CSVLogger(csv_path, append=False)
    
    callbacks = [checkpoint, early_stop, reduce_lr, csv_logger]
    
    # TensorBoard (optional)
    if log_dir:
        tensorboard_dir = os.path.join(log_dir, model_name)
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
    
    return callbacks


def train_model(model, train_generator, val_generator, model_name, epochs=None):
    if epochs is None:
        epochs = config.EPOCHS
    
    callbacks = get_callbacks(model_name)

    steps_per_epoch = train_generator.samples // config.BATCH_SIZE
    validation_steps = val_generator.samples // config.BATCH_SIZE

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def save_training_history(history, model_name):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    history_dict = {}
    for key, value in history.history.items():
        history_dict[key] = [float(v) for v in value]
    

    history_path = os.path.join(config.RESULTS_DIR, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)


def load_training_history(model_name):
    history_path = os.path.join(config.RESULTS_DIR, f'{model_name}_history.json')
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history_dict = json.load(f)
    
    return history_dict


def evaluate_model(model, generator):
    results = model.evaluate(generator, verbose=1)
    
    metrics = {}
    if isinstance(results, list):
        # TensorFlow 2.x returns a list
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = float(results[i])
    else:
        # Single value (loss only)
        metrics['loss'] = float(results)
    
    # Handle different metric naming conventions
    # Ensure 'accuracy' key exists (might be 'categorical_accuracy' or similar)
    if 'accuracy' not in metrics:
        for key in metrics.keys():
            if 'accuracy' in key.lower():
                metrics['accuracy'] = metrics[key]
                break
    
    return metrics


def get_predictions(model, generator):
    generator.reset()
    
    y_pred_proba = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    classes = generator.classes
    
    return classes, y_pred, y_pred_proba