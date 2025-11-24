import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import config


def plot_training_history(history_dict, model_name, save_dir=None):
    """
    Plot training history (loss and metrics)
    
    Args:
        history_dict: Dictionary containing training history
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
    
    # Plot loss
    axes[0, 0].plot(history_dict['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Model Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Model Accuracy', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot precision
    if 'precision' in history_dict:
        axes[1, 0].plot(history_dict['precision'], label='Training Precision', linewidth=2)
        axes[1, 0].plot(history_dict['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Model Precision', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot recall
    if 'recall' in history_dict:
        axes[1, 1].plot(history_dict['recall'], label='Training Recall', linewidth=2)
        axes[1, 1].plot(history_dict['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].set_title('Model Recall', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.SEVERITY_CLASS_NAMES,
                yticklabels=config.SEVERITY_CLASS_NAMES,
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_classification_report(y_true, y_pred, model_name, save_dir=None):
    """
    Generate and save classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save report
        
    Returns:
        str: Classification report
    """
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=config.SEVERITY_CLASS_NAMES,
        digits=4
    )
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Classification Report")
    print(f"{'='*60}")
    print(report)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, f'{model_name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"{model_name} - Classification Report\n")
            f.write("="*60 + "\n")
            f.write(report)
        print(f"Classification report saved to: {report_path}")
    
    return report


def plot_roc_curves(y_true, y_pred_proba, model_name, save_dir=None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=list(range(config.SEVERITY_NUM_CLASSES)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(config.SEVERITY_CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    
    return fig


def compare_models(model_results, save_dir=None):
    """
    Compare multiple models side by side
    
    Args:
        model_results: Dictionary with model names as keys and results as values
        save_dir: Directory to save comparison plots
    """
    model_names = list(model_results.keys())
    
    # Extract metrics
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'auc', 'loss']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_compare):
        values = []
        for model_name in model_names:
            if metric in model_results[model_name]:
                values.append(model_results[model_name][metric])
            else:
                values.append(0)
        
        # Create bar plot
        bars = axes[idx].bar(range(len(model_names)), values, color='steelblue', alpha=0.7)
        axes[idx].set_xticks(range(len(model_names)))
        axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].set_title(f'{metric.capitalize()} Comparison', fontsize=14)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.4f}',
                          ha='center', va='bottom', fontsize=10)
    
    # Training time comparison (if available)
    if any('training_time' in results for results in model_results.values()):
        training_times = []
        for model_name in model_names:
            if 'training_time' in model_results[model_name]:
                training_times.append(model_results[model_name]['training_time'])
            else:
                training_times.append(0)
        
        bars = axes[5].bar(range(len(model_names)), training_times, color='coral', alpha=0.7)
        axes[5].set_xticks(range(len(model_names)))
        axes[5].set_xticklabels(model_names, rotation=45, ha='right')
        axes[5].set_ylabel('Time (seconds)', fontsize=12)
        axes[5].set_title('Training Time Comparison', fontsize=14)
        axes[5].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, training_times):
            height = bar.get_height()
            axes[5].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}s',
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    return fig


def plot_all_confusion_matrices(model_results, save_dir=None):
    """
    Plot confusion matrices for all models in a grid
    
    Args:
        model_results: Dictionary with model names as keys and results as values
        save_dir: Directory to save plots
    """
    model_names = list(model_results.keys())
    n_models = len(model_names)
    
    # Calculate grid size
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        if 'confusion_matrix' in model_results[model_name]:
            cm = model_results[model_name]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=config.SEVERITY_CLASS_NAMES,
                       yticklabels=config.SEVERITY_CLASS_NAMES,
                       ax=axes[idx],
                       cbar=True)
            
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('True', fontsize=10)
            axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'all_confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All confusion matrices saved to: {save_path}")
    
    return fig


def plot_model_comparison_table(model_results, save_dir=None):
    """
    Create a comparison table of all models
    
    Args:
        model_results: Dictionary with model names as keys and results as values
        save_dir: Directory to save table
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for model_name, results in model_results.items():
        row = {
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'AUC': results.get('auc', 0)
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    for idx, row in df.iterrows():
        table_row = [
            row['Model'],
            f"{row['Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{row['AUC']:.4f}"
        ]
        table_data.append(table_row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style best model row
    for i in range(len(df.columns)):
        table[(1, i)].set_facecolor('#E8F5E9')
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model_comparison_table.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison table saved to: {save_path}")
        
        # Save as CSV
        csv_path = os.path.join(save_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"Model comparison CSV saved to: {csv_path}")
    
    return df, fig


def plot_per_class_performance(model_results, save_dir=None):
    """
    Plot per-class performance for all models
    
    Args:
        model_results: Dictionary with model names as keys and results as values
        save_dir: Directory to save plots
    """
    model_names = list(model_results.keys())
    class_names = config.SEVERITY_CLASS_NAMES
    
    # Extract per-class metrics
    metrics = ['precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Per-Class Performance Comparison', fontsize=16, fontweight='bold')
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Prepare data for grouped bar chart
        x = np.arange(len(class_names))
        width = 0.8 / len(model_names)
        
        for model_idx, model_name in enumerate(model_names):
            if 'per_class_metrics' in model_results[model_name]:
                values = []
                for class_name in class_names:
                    class_metrics = model_results[model_name]['per_class_metrics'].get(class_name, {})
                    values.append(class_metrics.get(metric, 0))
                
                offset = width * (model_idx - len(model_names)/2 + 0.5)
                ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Damage Severity Class', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} by Class', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'per_class_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class performance plot saved to: {save_path}")
    
    return fig


def create_comprehensive_report(model_results, save_dir=None):
    """
    Create a comprehensive report with all visualizations
    
    Args:
        model_results: Dictionary with model names as keys and results as values
        save_dir: Directory to save all plots
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    # 1. Model comparison
    print("\n1. Creating model comparison...")
    compare_models(model_results, save_dir)
    
    # 2. Comparison table
    print("\n2. Creating comparison table...")
    plot_model_comparison_table(model_results, save_dir)
    
    # 3. All confusion matrices
    print("\n3. Creating confusion matrices...")
    plot_all_confusion_matrices(model_results, save_dir)
    
    # 4. Per-class performance
    print("\n4. Creating per-class performance plots...")
    plot_per_class_performance(model_results, save_dir)
    
    print("\n" + "="*60)
    print(f"COMPREHENSIVE REPORT SAVED TO: {save_dir}")
    print("="*60 + "\n")
    
    plt.close('all')
    
    return fig
