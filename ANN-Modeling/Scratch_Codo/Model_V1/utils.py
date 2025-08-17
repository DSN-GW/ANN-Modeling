import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import json

def preprocess_data(data):
    text_column = 'free_response'
    
    numerical_columns = ['SRS.Raw', 'FSR', 'BIS', 'avg_PE', 
                         'LPA_Profile_grand_mean', 'LPA_Profile_ASD_only']
    
    target_column = 'td_or_asd'
    
    data = data.dropna(subset=[target_column])
    
    X_text = data[text_column].fillna('').tolist()
    
    numerical_data = data[numerical_columns].copy()
    numerical_data = numerical_data.fillna(numerical_data.mean())
    X_numerical = numerical_data.values
    
    y = data[target_column].values
    
    feature_names = {
        'text_column': text_column,
        'numerical_columns': numerical_columns
    }
    
    return X_text, X_numerical, y, feature_names

def plot_metrics(y_true, y_prob, y_pred, results_path):
    os.makedirs(results_path, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(results_path, 'roc_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(results_path, 'precision_recall_curve.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_path, 'confusion_matrix.png'))
    plt.close()

def save_results(metrics, feature_names, results_path):
    os.makedirs(results_path, exist_ok=True)
    
    results = {
        'accuracy': float(metrics['accuracy']),
        'f1_score': float(metrics['f1_score']),
        'auc': float(metrics['auc']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'features_used': feature_names
    }
    
    with open(os.path.join(results_path, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(os.path.join(results_path, 'results_summary.txt'), 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))
        f.write("\n\nFeatures Used:\n")
        f.write(f"Text Column: {feature_names['text_column']}\n")
        f.write(f"Numerical Columns: {', '.join(feature_names['numerical_columns'])}")