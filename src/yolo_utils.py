"""
Utilities for training and evaluating YOLOv8 classification model
"""
from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

from config import MODELS_DIR, RESULTS_DIR, YOLO_EPOCHS, YOLO_IMG_SIZE, CLASS_NAMES


def train_yolo_classifier(data_yaml_path, epochs=YOLO_EPOCHS, img_size=YOLO_IMG_SIZE, 
                          model_size='n', project_name='yolov8_damage'):
    """
    Train YOLOv8 classification model
    
    Args:
        data_yaml_path: Path to dataset.yaml file
        epochs: Number of training epochs
        img_size: Input image size
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        project_name: Project name for saving results
        
    Returns:
        Trained model and results
    """
    # Initialize model
    model = YOLO(f'yolov8{model_size}-cls.pt')
    
    print(f"Training YOLOv8{model_size.upper()} Classification Model...")
    print(f"Dataset: {data_yaml_path}")
    print(f"Epochs: {epochs}, Image Size: {img_size}")
    
    # Train model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=-1,  # Auto batch size
        project=str(MODELS_DIR / project_name),
        name='train',
        patience=20,
        save=True,
        plots=True,
        verbose=True
    )
    
    # Save model
    model_save_path = MODELS_DIR / f'{project_name}_best.pt'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, results


def evaluate_yolo_model(model, val_data_path):
    """
    Evaluate YOLOv8 classification model
    
    Args:
        model: Trained YOLO model or path to model
        val_data_path: Path to validation data
        
    Returns:
        Dictionary with evaluation metrics
    """
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    
    # Validate model
    results = model.val(data=val_data_path, split='val')
    
    # Extract metrics
    metrics = {
        'accuracy': float(results.results_dict.get('metrics/accuracy_top1', 0)),
        'top5_accuracy': float(results.results_dict.get('metrics/accuracy_top5', 0)),
    }
    
    print(f"\nYOLO Validation Results:")
    print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")
    print(f"  Accuracy (Top-5): {metrics['top5_accuracy']:.4f}")
    
    return metrics


def predict_yolo_batch(model, val_dir):
    """
    Make predictions on validation set and extract detailed metrics
    
    Args:
        model: Trained YOLO model or path to model
        val_dir: Path to validation directory
        
    Returns:
        Dictionary with predictions and metrics
    """
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    
    y_true = []
    y_pred = []
    
    # Process each class folder
    val_path = Path(val_dir)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = None
        
        # Find the class folder (handles different naming conventions)
        for folder in val_path.iterdir():
            if folder.is_dir() and class_name in folder.name.lower():
                class_folder = folder
                break
        
        if class_folder is None:
            # Try with index prefix
            for folder in val_path.iterdir():
                if folder.is_dir() and folder.name.startswith(f'0{class_idx+1}-'):
                    class_folder = folder
                    break
        
        if class_folder is None:
            print(f"Warning: Could not find folder for class {class_name}")
            continue
        
        # Get predictions for images in this class
        image_files = list(class_folder.glob('*'))
        
        for img_file in image_files:
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                results = model.predict(img_file, verbose=False)
                
                if len(results) > 0 and results[0].probs is not None:
                    pred_class = int(results[0].probs.top1)
                    y_true.append(class_idx)
                    y_pred.append(pred_class)
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
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
                'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                'f1_score': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
            }
            for i in range(len(CLASS_NAMES))
        },
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist()
    }
    
    return results


def save_yolo_results(results, model_name='yolov8'):
    """
    Save YOLO results to JSON file
    
    Args:
        results: Results dictionary
        model_name: Name of the model
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
    
    # Save to file
    save_path = MODELS_DIR / f'{model_name}_results.json'
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Saved YOLO results to {save_path}")
