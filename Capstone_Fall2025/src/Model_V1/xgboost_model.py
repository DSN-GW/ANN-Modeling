import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
import os
import json
from pathlib import Path

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
            # Get the project root directory (two levels up from current file)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            data_path = project_root / "data" / "Data_v1" / "LLM_data_train_preprocessed_v1.csv"
        
        df = pd.read_csv(data_path)
        return df
    
    def prepare_features(self, df):
        """
        Prepare features with careful feature selection to avoid target leakage.
        
        This method:
        1. Excludes direct identifiers and target variables
        2. Excludes potential leaking features (those with suspiciously high correlation with target)
        3. Applies feature selection to reduce dimensionality and prevent overfitting
        """
        # List of columns to exclude (direct identifiers, target, and text)
        exclude_columns = ['sub', 'profile', 'subject', 'td_or_asd', 'free_response']
        
        # Additional columns that might leak target information
        potential_leaking_columns = [
            'LPA_Profile_ASD_only',  # This directly relates to ASD diagnosis
            'SRS.Raw',               # Social Responsiveness Scale is used for ASD diagnosis
            # Add any other columns that might directly encode diagnosis information
        ]
        
        exclude_columns.extend(potential_leaking_columns)
        
        # Select numeric and boolean features, excluding the ones in exclude_columns
        feature_columns = []
        for col in df.columns:
            if col not in exclude_columns:
                if df[col].dtype in ['int64', 'float64', 'bool']:
                    feature_columns.append(col)
        
        # Create feature matrix and target vector
        X = df[feature_columns].copy()
        y = df['td_or_asd'].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        # Check for features with extremely high correlation with target
        # These might be leaking target information
        if len(X) > 1:  # Only if we have more than one sample
            correlations = []
            for col in X.columns:
                corr = abs(X[col].corr(y))
                correlations.append((col, corr))
            
            # Sort by correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Print warning for suspiciously high correlations
            suspicious_threshold = 0.9
            suspicious_features = [col for col, corr in correlations if corr > suspicious_threshold]
            
            if suspicious_features:
                print(f"WARNING: The following features have suspiciously high correlation (>{suspicious_threshold}) with the target:")
                for col, corr in correlations:
                    if corr > suspicious_threshold:
                        print(f"  - {col}: {corr:.4f}")
                print("Consider removing these features as they might be leaking target information.")
                
                # Remove the most suspicious features (optional)
                X = X.drop(columns=suspicious_features)
                feature_columns = [col for col in feature_columns if col not in suspicious_features]
                print(f"Removed {len(suspicious_features)} suspicious features.")
        
        # Store the final feature names
        self.feature_names = feature_columns
        
        print(f"Selected {len(feature_columns)} features after filtering.")
        return X, y
    
    def train_model(self, X, y, random_state=42):
        """
        Train the model on the entire training dataset without a second train-test split.
        Use cross-validation for model evaluation during training.
        
        Note: The test data is already separated in data_preprocessor.py, so we don't need
        another train-test split here. This prevents the double train-test split issue.
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create a more regularized model to prevent overfitting
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,  # Reduced from 6 to prevent overfitting
            learning_rate=0.05,  # Reduced from 0.1 to prevent overfitting
            subsample=0.7,  # Reduced from 0.8 for more regularization
            colsample_bytree=0.7,  # Reduced from 0.8 for more regularization
            min_child_weight=3,  # Added to prevent overfitting
            gamma=0.1,  # Added to prevent overfitting
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=random_state,
            eval_metric='logloss'
        )
        
        # Use stratified k-fold cross-validation for more reliable evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Train the final model on the entire training set
        self.model.fit(X_scaled, y)
        
        # Get feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Create results dictionary with cross-validation metrics
        results = {
            'feature_importance': self.feature_importance,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'model_params': self.model.get_params()
        }
        
        # Note: We don't calculate accuracy, classification_report, etc. here
        # because we're not doing a second train-test split. These metrics
        # will be calculated on the true test set in predict.py.
        
        return results
    
    def get_feature_importance_sorted(self, top_n=20):
        if self.feature_importance is None:
            return None
        
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]
    
    def save_model(self, model_dir=None):
        if model_dir is None:
            # Get the project root directory (two levels up from current file)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            model_dir = project_root / "Results" / "V1"
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / 'xgboost_model_v1.pkl'
        scaler_path = model_dir / 'scaler_v1.pkl'
        features_path = model_dir / 'feature_names_v1.pkl'
        importance_path = model_dir / 'feature_importance_v1.json'
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        with open(importance_path, 'w') as f:
            json.dump(convert_numpy_types(self.feature_importance), f, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir=None):
        if model_dir is None:
            # Get the project root directory (two levels up from current file)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            model_dir = project_root / "Results" / "V1"
        
        model_dir = Path(model_dir)
        model_path = model_dir / 'xgboost_model_v1.pkl'
        scaler_path = model_dir / 'scaler_v1.pkl'
        features_path = model_dir / 'feature_names_v1.pkl'
        importance_path = model_dir / 'feature_importance_v1.json'
        
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
    
    print("Training XGBoost model with cross-validation...")
    results = classifier.train_model(X, y)
    
    print("Training Results:")
    print(f"Cross-validation mean accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    print(f"Cross-validation scores: {[f'{score:.4f}' for score in results['cv_scores']]}")
    
    print("\nModel Parameters:")
    for param, value in results['model_params'].items():
        if param in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree', 
                    'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']:
            print(f"  {param}: {value}")
    
    print("\nTop 10 Most Important Features:")
    top_features = classifier.get_feature_importance_sorted(10)
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")
    
    print("Saving model...")
    classifier.save_model()
    
    # Get the project root directory (two levels up from current file)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    results_path = project_root / "Results" / "V1" / "training_results_v1.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    print("\nNote: Final model evaluation will be performed on the test set in predict.py")
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = train_xgboost_model()