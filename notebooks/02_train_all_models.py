import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
from data_loader import create_data_generators, get_dataset_info
from models import get_model, compile_model, get_model_summary
from train import train_model, save_training_history, evaluate_model, get_predictions
from visualize import (
    plot_training_history, 
    plot_confusion_matrix, 
    plot_classification_report,
    plot_roc_curves
)
from sklearn.metrics import confusion_matrix

np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)

train_gen, val_gen = create_data_generators()
dataset_info = get_dataset_info(train_gen, val_gen)
all_results = {}

for i, architecture in enumerate(config.SEVERITY_MODEL_ARCHITECTURES, 1):
    try:
        model = get_model(architecture)
        model = compile_model(model)
        model.summary()

        summary_path = os.path.join(config.RESULTS_DIR, f'{architecture}_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Train model
        start_time = time.time()
        history = train_model(
            model=model,
            train_generator=train_gen,
            val_generator=val_gen,
            model_name=architecture,
            epochs=config.EPOCHS
        )
        training_time = time.time() - start_time
        
        print(f"\n{architecture} training completed in {training_time:.2f} seconds!")
        save_training_history(history, architecture)
        best_model_path = os.path.join(config.MODELS_DIR, f'{architecture}_best.h5')
        model = keras.models.load_model(best_model_path)

        val_metrics = evaluate_model(model, val_gen)
        y_true, y_pred, y_pred_proba = get_predictions(model, val_gen)

        cm = confusion_matrix(y_true, y_pred)
        all_results[architecture] = {
            'accuracy': val_metrics['accuracy'],
            'loss': val_metrics['loss'],
            'precision': val_metrics.get('precision', 0),
            'recall': val_metrics.get('recall', 0),
            'auc': val_metrics.get('auc', 0),
            'training_time': training_time,
            'confusion_matrix': cm.tolist(),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        # Plot training history
        plot_training_history(
            history.history, 
            architecture, 
            save_dir=config.RESULTS_DIR
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true, 
            y_pred, 
            architecture, 
            save_dir=config.RESULTS_DIR
        )
        
        # Plot classification report
        plot_classification_report(
            y_true, 
            y_pred, 
            architecture, 
            save_dir=config.RESULTS_DIR
        )
        
        # Plot ROC curves
        plot_roc_curves(
            y_true, 
            y_pred_proba, 
            architecture, 
            save_dir=config.RESULTS_DIR
        )
        
        print(f"\n {architecture} training and evaluation complete!")
        
        # Clear session to free memory
        keras.backend.clear_session()
        
    except Exception as e:
        print(f"\n Error training {architecture}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Save all results
results_path = os.path.join(config.RESULTS_DIR, 'all_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4)