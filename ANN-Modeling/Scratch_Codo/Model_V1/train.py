import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from utils import preprocess_data
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

def create_weighted_sampler(labels):
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts, numerical)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs.squeeze()) >= 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, accuracy, f1

def main():
    data_path = os.path.join('../../Code', '..', 'data', 'Data_v1', 'LLM data.csv')
    results_path = os.path.join('../../Code', '..', 'Results', 'V1')
    
    data = load_data(data_path)
    
    X_text, X_numerical, y, feature_names = preprocess_data(data)
    
    X_text_train, X_text_val, X_numerical_train, X_numerical_val, y_train, y_val = train_test_split(
        X_text, X_numerical, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_numerical_train = scaler.fit_transform(X_numerical_train)
    X_numerical_val = scaler.transform(X_numerical_val)
    
    train_dataset = AutismDataset(X_text_train, X_numerical_train, y_train)
    val_dataset = AutismDataset(X_text_val, X_numerical_val, y_val)
    
    train_sampler = create_weighted_sampler(y_train)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AutismClassifier(
        num_numerical_features=X_numerical_train.shape[1],
        device=device
    )
    
    model.to(device)
    
    num_epochs = 20
    learning_rate = 2e-5
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_train_f1 = 0.0
    best_model_state = None
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution - Val: {np.bincount(y_val.astype(int))}")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            texts = batch['text']
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(texts, numerical)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        train_loss, train_accuracy, train_f1 = evaluate_model(model, train_loader, criterion, device)
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            best_model_state = model.state_dict().copy()
            print(f'  New best training F1 score: {best_train_f1:.4f}')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with training F1 score: {best_train_f1:.4f}')
    
    os.makedirs(results_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(results_path, 'model.pt'))
    
    with open(os.path.join(results_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    train_indices = list(range(len(X_text_train)))
    val_indices = list(range(len(X_text_train), len(X_text_train) + len(X_text_val)))
    
    with open(os.path.join(results_path, 'train_indices.pkl'), 'wb') as f:
        pickle.dump(train_indices, f)
    
    with open(os.path.join(results_path, 'val_indices.pkl'), 'wb') as f:
        pickle.dump(val_indices, f)
    
    with open(os.path.join(results_path, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Training completed and model saved!")

if __name__ == '__main__':
    main()