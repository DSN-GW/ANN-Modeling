import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost_model import XGBoostClassifier
import os
import json

class ExplainabilityAnalyzer:
    def __init__(self):
        self.classifier = XGBoostClassifier()
        self.explainer = None
        self.shap_values = None
        self.characteristics = self.load_characteristics()
        
    def load_characteristics(self):
        char_path = os.path.join('..', '..', 'data', 'Data_v1', 'charactristic.txt')
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
            char_features = [col for col in X.columns if col.startswith(char)]
            
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
            char_features = [col for col in X.columns if col.startswith(char)]
            
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
                'feature_count': len(self.classifier.feature_names),
                'top_features': self.get_top_discriminative_features(15)
            },
            'characteristic_analysis': self.analyze_characteristic_contributions(X, y),
            'class_patterns': {}
        }
        
        td_patterns, asd_patterns = self.analyze_td_vs_asd_patterns(None, X, y)
        report['class_patterns']['td_patterns'] = td_patterns
        report['class_patterns']['asd_patterns'] = asd_patterns
        
        return report
    
    def save_explainability_results(self, report, text_analysis):
        results_dir = os.path.join('..', '..', 'Results', 'V1')
        os.makedirs(results_dir, exist_ok=True)
        
        explainability_path = os.path.join(results_dir, 'explainability_analysis_v1.json')
        text_analysis_path = os.path.join(results_dir, 'text_patterns_analysis_v1.json')
        
        with open(explainability_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        with open(text_analysis_path, 'w') as f:
            json.dump(text_analysis, f, indent=2)
        
        print(f"Explainability results saved to {results_dir}")
        
        return explainability_path, text_analysis_path
    
    def create_characteristic_summary(self, report):
        char_summary = {}
        
        for char, data in report['characteristic_analysis'].items():
            char_summary[char] = {
                'importance_score': data['total_importance'],
                'feature_count': data['feature_count'],
                'rank': 0
            }
        
        sorted_chars = sorted(char_summary.items(), key=lambda x: x[1]['importance_score'], reverse=True)
        
        for rank, (char, data) in enumerate(sorted_chars, 1):
            char_summary[char]['rank'] = rank
        
        return char_summary
    
    def analyze_prediction_explanations(self, df, X, y, sample_size=100):
        sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        shap_values = self.initialize_explainer(X_sample)
        
        explanations = []
        for i, idx in enumerate(sample_indices):
            explanation = {
                'sample_id': int(idx),
                'true_label': int(y_sample.iloc[i]),
                'predicted_label': int(self.classifier.model.predict(self.classifier.scaler.transform(X_sample.iloc[i:i+1]))[0]),
                'top_contributing_features': {}
            }
            
            feature_contributions = dict(zip(X.columns, shap_values[i]))
            top_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            
            for feature, contribution in top_features:
                explanation['top_contributing_features'][feature] = float(contribution)
            
            explanations.append(explanation)
        
        return explanations

def run_explainability_analysis():
    print("Initializing explainability analyzer...")
    analyzer = ExplainabilityAnalyzer()
    
    print("Loading model and data...")
    df, X, y = analyzer.load_model_and_data()
    
    print("Generating feature contribution report...")
    report = analyzer.generate_feature_contribution_report(X, y)
    
    print("Analyzing text patterns by class...")
    text_analysis = analyzer.analyze_text_patterns_by_class(df)
    
    print("Creating characteristic summary...")
    char_summary = analyzer.create_characteristic_summary(report)
    
    print("Analyzing prediction explanations...")
    explanations = analyzer.analyze_prediction_explanations(df, X, y)
    
    report['characteristic_summary'] = char_summary
    report['prediction_explanations'] = explanations
    
    print("Saving results...")
    explainability_path, text_path = analyzer.save_explainability_results(report, text_analysis)
    
    print("\nTop 5 Most Important Characteristics:")
    for char, data in sorted(char_summary.items(), key=lambda x: x[1]['importance_score'], reverse=True)[:5]:
        print(f"{data['rank']}. {char}: {data['importance_score']:.4f} (features: {data['feature_count']})")
    
    return analyzer, report, text_analysis

if __name__ == "__main__":
    analyzer, report, text_analysis = run_explainability_analysis()