"""
Training utilities for damage severity classification models
Includes training loops, callbacks, metrics tracking, and evaluation
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import config


def compile_model(model, learning_rate=None):
    """
    Compile model with optimizer, loss, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    """
    if learning_rate is None:
        learning_rate = config.INITIAL_LEARNING_RATE
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print(f"Model compiled with learning rate: {learning_rate}")
    return model


def get_callbacks(model_name, stage='initial'):
    """
    Create callbacks for training
    
    Args:
        model_name: Name of the model
        stage: 'initial' or 'finetune'
        
    Returns:
        List of callbacks
    """
    # Create directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Model checkpoint path
    checkpoint_path = os.path.join(
        config.MODELS_DIR, 
        f'{model_name}_{stage}_{timestamp}.h5'
    )
    
    # CSV logger path
    csv_path = os.path.join(
        config.RESULTS_DIR,
        f'{model_name}_{stage}_training_{timestamp}.csv'
    )
    
    # TensorBoard log path
    tensorboard_path = os.path.join(
        config.LOGS_DIR,
        f'{model_name}_{stage}_{timestamp}'
    )
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV logging
        CSVLogger(
            filename=csv_path,
            separator=',',
            append=False
        )
    ]
    
    print(f"\nCallbacks configured:")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - CSV Log: {csv_path}")
    print(f"  - TensorBoard: {tensorboard_path}\n")
    
    return callbacks, checkpoint_path


def train_model(model, train_dataset, val_dataset, model_name, epochs=None, callbacks_list=None):
    """
    Train the model
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_name: Name of the model
        epochs: Number of epochs
        callbacks_list: List of callbacks
        
    Returns:
        history: Training history
        checkpoint_path: Path to saved model
    """
    if epochs is None:
        epochs = config.EPOCHS
    
    if callbacks_list is None:
        callbacks_list, checkpoint_path = get_callbacks(model_name, 'initial')
    else:
        checkpoint_path = None
        for cb in callbacks_list:
            if isinstance(cb, ModelCheckpoint):
                checkpoint_path = cb.filepath
                break
    
    print(f"\n{'='*60}")
    print(f"Starting training for {model_name}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed for {model_name}")
    print(f"{'='*60}\n")
    
    return history, checkpoint_path


def evaluate_model(model, test_dataset, model_name):
    """
    Evaluate model on test/validation dataset
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    results = model.evaluate(test_dataset, verbose=1)
    
    # Get metric names
    metric_names = model.metrics_names
    
    # Create results dictionary
    eval_results = {name: float(value) for name, value in zip(metric_names, results)}
    
    print(f"\nEvaluation Results for {model_name}:")
    print("-" * 40)
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    print(f"{'='*60}\n")
    
    return eval_results


def get_predictions(model, dataset, class_names=None):
    """
    Get predictions and true labels from dataset
    
    Args:
        model: Trained model
        dataset: Dataset to predict on
        class_names: List of class names
        
    Returns:
        y_true, y_pred, y_pred_proba
    """
    if class_names is None:
        class_names = config.SEVERITY_CLASS_NAMES
    
    y_true = []
    y_pred_proba = []
    
    print("Generating predictions...")
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_true, y_pred, y_pred_proba


def calculate_metrics(y_true, y_pred, y_pred_proba, class_names=None):
    """
    Calculate detailed classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support
    )
    
    if class_names is None:
        class_names = config.SEVERITY_CLASS_NAMES
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            for i in range(len(class_names))
        }
    }
    
    return metrics


def save_training_results(model_name, history, eval_metrics, checkpoint_path):
    """
    Save training results to JSON file
    
    Args:
        model_name: Name of the model
        history: Training history
        eval_metrics: Evaluation metrics
        checkpoint_path: Path to saved model
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(
        config.RESULTS_DIR,
        f'{model_name}_results_{timestamp}.json'
    )
    
    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'checkpoint_path': checkpoint_path,
        'training_history': {
            key: [float(v) for v in value] 
            for key, value in history.history.items()
        },
        'evaluation_metrics': eval_metrics,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'initial_lr': config.INITIAL_LEARNING_RATE
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_file


def load_training_history(csv_path):
    """Load training history from CSV file"""
    return pd.read_csv(csv_path)


def compare_models(results_dict):
    """
    Compare multiple models and create summary
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy', ascending=False)
    
    return df
