"""
Visualization utilities for model training and evaluation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path

from config import CLASS_NAMES, RESULTS_DIR


def plot_training_history(history, model_name, save=True):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Keras history object or dict with keys: loss, accuracy, val_loss, val_accuracy
        model_name: Name of the model
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Check if history is a Keras History object or dict
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    # Plot accuracy
    axes[0].plot(history_dict['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history_dict['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history_dict['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history_dict['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_path = RESULTS_DIR / f'{model_name}_training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name, save=True):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save: Whether to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save:
        save_path = RESULTS_DIR / f'{model_name}_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return plt.gcf()


def print_classification_report(y_true, y_pred, model_name):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Classification Report - {model_name}")
    print(f"{'='*60}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(f"{'='*60}\n")


def compare_models(results_dict, save=True):
    """
    Compare multiple models' performance
    
    Args:
        results_dict: Dictionary with model_name as keys and metrics dict as values
                      e.g., {'YOLOv8': {'accuracy': 0.95, 'precision': 0.94, ...}}
        save: Whether to save the plot
    """
    df = pd.DataFrame(results_dict).T
    
    # Create subplots for different metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[idx]
            bars = ax.bar(df.index, df[metric], color=plt.cm.viridis(np.linspace(0, 1, len(df))))
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Rotate x labels if needed
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save:
        save_path = RESULTS_DIR / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot to {save_path}")
    
    return fig


def plot_prediction_samples(images, true_labels, pred_labels, class_names=CLASS_NAMES, num_samples=9):
    """
    Plot sample predictions with true and predicted labels
    
    Args:
        images: Array of images
        true_labels: True class indices
        pred_labels: Predicted class indices
        class_names: List of class names
        num_samples: Number of samples to display
    """
    num_samples = min(num_samples, len(images))
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(num_samples):
        ax = axes[idx]
        img = images[idx]
        
        # Denormalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[pred_labels[idx]]
        
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        ax.set_title(f'True: {true_class}\nPred: {pred_class}', 
                    color=color, fontweight='bold', fontsize=12)
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_class_distribution(train_dist, val_dist, save=True):
    """
    Plot class distribution for training and validation sets
    
    Args:
        train_dist: Dictionary with class distribution for training
        val_dist: Dictionary with class distribution for validation
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training distribution
    axes[0].bar(train_dist.keys(), train_dist.values(), color='skyblue', edgecolor='black')
    axes[0].set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Images')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (k, v) in enumerate(train_dist.items()):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Validation distribution
    axes[1].bar(val_dist.keys(), val_dist.values(), color='lightcoral', edgecolor='black')
    axes[1].set_title('Validation Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Images')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (k, v) in enumerate(val_dist.items()):
        axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        save_path = RESULTS_DIR / 'class_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution plot to {save_path}")
    
    return fig
