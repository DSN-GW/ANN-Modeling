import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
import os
import json
from pathlib import Path

data_version = "Data_v7-3"
model_version = "V7-3"

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class XGBoostClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
    def load_data(self, data_path=None):
        if data_path is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            data_path = project_root / "data" / data_version / f"LLM_data_train_preprocessed_{model_version}.csv"
        
        df = pd.read_csv(data_path)
        return df
    
    def prepare_features(self, df):
        exclude_columns = ['sub', 'profile', 'FSR_scaled','avg_PE_scaled', 'free_response', 'td_or_asd']
        
        feature_columns = []
        for col in df.columns:
            if col not in exclude_columns:
                if df[col].dtype in ['int64', 'float64', 'bool']:
                    feature_columns.append(col)
        
        X = df[feature_columns].copy()
        y = df['td_or_asd'].copy()
        
        X = X.fillna(0)
        
        self.feature_names = feature_columns
        
        print(f"Selected {len(feature_columns)} features.")
        return X, y
    
    def train_model(self, X, y, random_state=42):
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        self.model.fit(X_scaled, y)
        
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring='accuracy')
        
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        training_results = {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        return training_results
    
    def get_feature_importance_sorted(self, top_n=20):
        if self.feature_importance is None:
            return None
        return sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def save_model(self, model_dir=None):
        if model_dir is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            model_dir = project_root / "Results" / model_version
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / f'xgboost_model_{model_version}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(model_dir / f'scaler_{model_version}.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / f'feature_names_{model_version}.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        if self.feature_importance:
            feature_importance_serializable = convert_numpy_types(self.feature_importance)
            with open(model_dir / f'feature_importance_{model_version}.json', 'w') as f:
                json.dump(feature_importance_serializable, f, indent=2)
        
        print(f"Model artifacts saved to: {model_dir}")
    
    def load_model(self, model_dir=None):
        if model_dir is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            model_dir = project_root / "Results" / model_version
        
        model_dir = Path(model_dir)
        
        with open(model_dir / f'xgboost_model_{model_version}.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(model_dir / f'scaler_{model_version}.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(model_dir / f'feature_names_{model_version}.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        feature_importance_path = model_dir / f'feature_importance_{model_version}.json'
        if feature_importance_path.exists():
            with open(feature_importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        print(f"Model loaded from: {model_dir}")
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

def train_xgboost_model():
    classifier = XGBoostClassifier()
    
    df = classifier.load_data()
    print(f"Loaded training data with shape: {df.shape}")
    
    X, y = classifier.prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    training_results = classifier.train_model(X, y)
    
    classifier.save_model()
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    results_dir = project_root / "Results" / f"{model_version}"
    
    training_results_serializable = convert_numpy_types(training_results)
    with open(results_dir / f'training_results_{model_version}.json', 'w') as f:
        json.dump(training_results_serializable, f, indent=2)
    
    print(f"Training completed successfully!")
    print(f"Cross-validation accuracy: {training_results['cv_accuracy_mean']:.4f} (+/- {training_results['cv_accuracy_std']:.4f})")
    
    return classifier, training_results

if __name__ == "__main__":
    classifier, results = train_xgboost_model()