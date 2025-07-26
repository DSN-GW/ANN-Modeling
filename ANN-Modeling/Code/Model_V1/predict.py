import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import preprocess_data, save_results, plot_metrics
from classes.model import AutismClassifier

class AutismDataset(Dataset):
    def __init__(self, texts, numerical_features, labels):
        self.texts = texts
        self.numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'numerical': self.numerical_features[idx],
            'label': self.labels[idx]
        }

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def main():
    data_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM data.csv')
    results_path = os.path.join('..', '..', 'Results', 'V1')
    
    data = load_data(data_path)
    
    X_text, X_numerical, y, feature_names = preprocess_data(data)
    
    X_text_train, X_text_val, X_numerical_train, X_numerical_val, y_train, y_val = train_test_split(
        X_text, X_numerical, y, test_size=0.2, random_state=42, stratify=y
    )
    
    with open(os.path.join(results_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    X_numerical_val = scaler.transform(X_numerical_val)
    
    val_dataset = AutismDataset(X_text_val, X_numerical_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AutismClassifier(
        num_numerical_features=X_numerical_val.shape[1],
        device=device
    )

    model.load_state_dict(torch.load(os.path.join(results_path, 'model.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Class distribution - Val: {np.bincount(y_val.astype(int))}")
    
    val_preds = []
    val_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            texts = batch['text']
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts, numerical)
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            
            # Handle single sample case
            if probs.ndim == 0:
                probs = np.array([probs])
            
            val_probs.extend(probs)
            val_preds.extend((probs >= 0.5).astype(int))
    
    accuracy = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_probs)
    conf_matrix = confusion_matrix(y_val, val_preds)
    
    print(f'\n=== VALIDATION RESULTS ===')
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation F1 Score: {f1:.4f}')
    print(f'Validation AUC: {auc:.4f}')
    print(f'Confusion Matrix:')
    print(conf_matrix)
    
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f'\n=== DETAILED METRICS ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall (Sensitivity): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'confusion_matrix': conf_matrix
    }
    
    plot_metrics(y_val, val_probs, val_preds, results_path)
    save_results(metrics, feature_names, results_path)
    
    print("\nPrediction completed and results saved!")

if __name__ == '__main__':
    main()