import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
import os
import json

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
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
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.feature_importance = None
        
    def load_data(self, data_path=None):
        if data_path is None:
            # Use the new training data path with stratified split
            data_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_train_preprocessed.csv')
        
        df = pd.read_csv(data_path)
        return df
    
    def prepare_features(self, df):
        feature_columns = []
        
        for col in df.columns:
            if col not in ['sub', 'profile', 'subject', 'td_or_asd', 'free_response']:
                if df[col].dtype in ['int64', 'float64', 'bool']:
                    feature_columns.append(col)
        
        X = df[feature_columns].copy()
        y = df['td_or_asd'].copy()
        
        X = X.fillna(0)
        
        self.feature_names = feature_columns
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self.feature_importance
        }
        
        if len(np.unique(y)) == 2:
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        return results, X_test, y_test, y_pred, y_pred_proba
    
    def get_feature_importance_sorted(self, top_n=20):
        if self.feature_importance is None:
            return None
        
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]
    
    def save_model(self, model_dir='../../Results/V2'):
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'xgboost_model_v2.pkl')
        scaler_path = os.path.join(model_dir, 'scaler_v2.pkl')
        features_path = os.path.join(model_dir, 'feature_names_v2.pkl')
        importance_path = os.path.join(model_dir, 'feature_importance_v2.json')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        with open(importance_path, 'w') as f:
            json.dump(convert_numpy_types(self.feature_importance), f, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='../../Results/V2'):
        model_path = os.path.join(model_dir, 'xgboost_model_v2.pkl')
        scaler_path = os.path.join(model_dir, 'scaler_v2.pkl')
        features_path = os.path.join(model_dir, 'feature_names_v2.pkl')
        importance_path = os.path.join(model_dir, 'feature_importance_v2.json')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        with open(importance_path, 'r') as f:
            self.feature_importance = json.load(f)
        
        print(f"Model loaded from {model_dir}")
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

def train_xgboost_model():
    print("Initializing XGBoost classifier...")
    classifier = XGBoostClassifier()
    
    print("Loading preprocessed data...")
    df = classifier.load_data()
    
    print("Preparing features...")
    X, y = classifier.prepare_features(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(classifier.feature_names)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    print("Training XGBoost model...")
    results, X_test, y_test, y_pred, y_pred_proba = classifier.train_model(X, y)
    
    print("Training Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation mean: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    if 'roc_auc' in results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    print("\nTop 10 Most Important Features:")
    top_features = classifier.get_feature_importance_sorted(10)
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")
    
    print("Saving model...")
    classifier.save_model()
    
    results_path = os.path.join('..', '..', 'Results', 'V2', 'training_results_v2.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = train_xgboost_model()