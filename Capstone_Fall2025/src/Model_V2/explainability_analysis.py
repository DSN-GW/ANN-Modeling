import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost_model import XGBoostClassifier
import os
import json
from pathlib import Path

data_version = "Data_v2"
model_version = "V2"

class ExplainabilityAnalyzer:
    def __init__(self):
        self.classifier = XGBoostClassifier()
        self.explainer = None
        self.shap_values = None
        self.characteristics = self.load_characteristics()
        
    def load_characteristics(self):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        char_path = project_root / "data" / data_version / "charactristic.txt"
        with open(char_path, 'r') as f:
            characteristics = [line.strip() for line in f.readlines() if line.strip()]
        return characteristics
    
    def load_model_and_data(self):
        self.classifier.load_model()
        df = self.classifier.load_data()
        X, y = self.classifier.prepare_features(df)
        return df, X, y
    
    def initialize_explainer(self, X_sample):
        X_sample_scaled = self.classifier.scaler.transform(X_sample)
        self.explainer = shap.TreeExplainer(self.classifier.model)
        self.shap_values = self.explainer.shap_values(X_sample_scaled)
        return self.shap_values
    
    def analyze_characteristic_contributions(self, X, y):
        characteristic_contributions = {}
        
        for char in self.characteristics:
            char_clean = char.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            char_features = [col for col in X.columns if col.startswith(char_clean)]
            
            if char_features:
                char_importance = 0
                char_shap_values = []
                
                for feature in char_features:
                    if feature in self.classifier.feature_importance:
                        char_importance += self.classifier.feature_importance[feature]
                
                characteristic_contributions[char] = {
                    'total_importance': char_importance,
                    'features': char_features,
                    'feature_count': len(char_features)
                }
        
        return characteristic_contributions
    
    def analyze_td_vs_asd_patterns(self, df, X, y):
        td_patterns = {}
        asd_patterns = {}
        
        td_mask = y == 0
        asd_mask = y == 1
        
        for char in self.characteristics:
            char_clean = char.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            char_features = [col for col in X.columns if col.startswith(char_clean)]
            
            if char_features:
                td_values = X[td_mask][char_features].mean()
                asd_values = X[asd_mask][char_features].mean()
                
                td_patterns[char] = td_values.to_dict()
                asd_patterns[char] = asd_values.to_dict()
        
        return td_patterns, asd_patterns
    
    def get_top_discriminative_features(self, n_features=20):
        if self.classifier.feature_importance is None:
            return None
        
        sorted_features = sorted(
            self.classifier.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:n_features]
    
    def analyze_text_patterns_by_class(self, df):
        text_analysis = {}
        
        for class_val in df['td_or_asd'].unique():
            class_texts = df[df['td_or_asd'] == class_val]['free_response'].dropna()
            
            all_text = ' '.join(class_texts.astype(str))
            words = all_text.lower().split()
            
            from collections import Counter
            word_counts = Counter(words)
            
            text_analysis[f'class_{class_val}'] = {
                'total_texts': len(class_texts),
                'total_words': len(words),
                'unique_words': len(set(words)),
                'top_words': dict(word_counts.most_common(20)),
                'avg_text_length': class_texts.str.len().mean()
            }
        
        return text_analysis
    
    def generate_feature_contribution_report(self, X, y):
        report = {
            'model_performance': {
                'total_features': len(X.columns),
                'sample_size': len(X)
            },
            'characteristic_analysis': self.analyze_characteristic_contributions(X, y),
            'top_features': self.get_top_discriminative_features(),
            'td_vs_asd_patterns': self.analyze_td_vs_asd_patterns(None, X, y)
        }
        
        return report
    
    def save_analysis_results(self, analysis_results, filename=f'explainability_analysis_{model_version}.json'):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        results_dir = project_root / "Results" / model_version
        results_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        analysis_results_serializable = convert_numpy_types(analysis_results)
        
        save_path = results_dir / filename
        with open(save_path, 'w') as f:
            json.dump(analysis_results_serializable, f, indent=2)
        
        print(f"Analysis results saved to: {save_path}")

def run_explainability_analysis():
    analyzer = ExplainabilityAnalyzer()
    
    print("Loading model and data...")
    df, X, y = analyzer.load_model_and_data()
    
    print("Generating explainability analysis...")
    analysis_results = analyzer.generate_feature_contribution_report(X, y)
    
    print("Analyzing text patterns...")
    text_patterns = analyzer.analyze_text_patterns_by_class(df)
    analysis_results['text_patterns'] = text_patterns
    
    print("Saving analysis results...")
    analyzer.save_analysis_results(analysis_results)
    
    print("Explainability analysis completed successfully!")
    return analysis_results

if __name__ == "__main__":
    results = run_explainability_analysis()