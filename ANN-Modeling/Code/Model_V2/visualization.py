import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from xgboost_model import XGBoostClassifier
from explainability_analysis import ExplainabilityAnalyzer

class ModelVisualizer:
    def __init__(self):
        self.results_dir = os.path.join('..', '..', 'Results', 'V2')
        self.viz_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_results(self):
        explainability_path = os.path.join(self.results_dir, 'explainability_analysis_v2.json')
        training_results_path = os.path.join(self.results_dir, 'training_results_v2.json')
        
        with open(explainability_path, 'r') as f:
            explainability_data = json.load(f)
        
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        
        return explainability_data, training_results
    
    def create_feature_importance_plot(self, training_results):
        feature_importance = training_results['feature_importance']
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features - XGBoost Model V2')
        plt.gca().invert_yaxis()
        
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importances[i]:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'feature_importance_v2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_characteristic_importance_stacked_bar(self, explainability_data):
        char_summary = explainability_data['characteristic_summary']
        char_analysis = explainability_data['characteristic_analysis']
        
        characteristics = []
        importance_scores = []
        feature_counts = []
        
        for char, data in sorted(char_summary.items(), key=lambda x: x[1]['importance_score'], reverse=True):
            characteristics.append(char.replace('_', ' ').title())
            importance_scores.append(data['importance_score'])
            feature_counts.append(data['feature_count'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        bars1 = ax1.bar(characteristics, importance_scores, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Total Importance Score')
        ax1.set_title('Characteristic Importance Scores - Model V2')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars1, importance_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        bars2 = ax2.bar(characteristics, feature_counts, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Number of Features')
        ax2.set_xlabel('Characteristics')
        ax2.set_title('Feature Count by Characteristic')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars2, feature_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'characteristic_importance_v2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_td_vs_asd_comparison_plot(self, explainability_data):
        td_patterns = explainability_data['class_patterns']['td_patterns']
        asd_patterns = explainability_data['class_patterns']['asd_patterns']
        
        characteristics = list(td_patterns.keys())
        
        td_mentioned = []
        asd_mentioned = []
        td_sentiment = []
        asd_sentiment = []
        
        for char in characteristics:
            td_char_data = td_patterns[char]
            asd_char_data = asd_patterns[char]
            
            td_mentioned.append(td_char_data.get(f'{char}_mentioned', 0))
            asd_mentioned.append(asd_char_data.get(f'{char}_mentioned', 0))
            td_sentiment.append(td_char_data.get(f'{char}_sentiment', 0))
            asd_sentiment.append(asd_char_data.get(f'{char}_sentiment', 0))
        
        x = np.arange(len(characteristics))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        bars1 = ax1.bar(x - width/2, td_mentioned, width, label='TD', color='lightblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, asd_mentioned, width, label='ASD', color='lightcoral', alpha=0.8)
        
        ax1.set_ylabel('Average Mention Rate')
        ax1.set_title('Characteristic Mention Rates: TD vs ASD - Model V2')
        ax1.set_xticks(x)
        ax1.set_xticklabels([char.replace('_', ' ').title() for char in characteristics], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        bars3 = ax2.bar(x - width/2, td_sentiment, width, label='TD', color='lightgreen', alpha=0.8)
        bars4 = ax2.bar(x + width/2, asd_sentiment, width, label='ASD', color='orange', alpha=0.8)
        
        ax2.set_ylabel('Average Sentiment Score')
        ax2.set_xlabel('Characteristics')
        ax2.set_title('Characteristic Sentiment Scores: TD vs ASD')
        ax2.set_xticks(x)
        ax2.set_xticklabels([char.replace('_', ' ').title() for char in characteristics], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'td_vs_asd_comparison_v2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_performance_plot(self, training_results):
        metrics = training_results['classification_report']
        
        classes = ['0', '1']
        precision = [metrics[cls]['precision'] for cls in classes]
        recall = [metrics[cls]['recall'] for cls in classes]
        f1_score = [metrics[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics by Class - XGBoost V2')
        ax.set_xticks(x)
        ax.set_xticklabels(['TD (Class 0)', 'ASD (Class 1)'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        accuracy = training_results['accuracy']
        cv_mean = training_results['cv_mean']
        cv_std = training_results['cv_std']
        
        ax.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.3f}\nCV Mean: {cv_mean:.3f} ± {cv_std:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'model_performance_v2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrix_plot(self, training_results):
        cm = np.array(training_results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['TD', 'ASD'], yticklabels=['TD', 'ASD'])
        plt.title('Confusion Matrix - XGBoost Model V2')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'confusion_matrix_v2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_characteristic_ranking_plot(self, explainability_data):
        char_summary = explainability_data['characteristic_summary']
        
        sorted_chars = sorted(char_summary.items(), key=lambda x: x[1]['importance_score'], reverse=True)
        
        characteristics = [char.replace('_', ' ').title() for char, _ in sorted_chars]
        scores = [data['importance_score'] for _, data in sorted_chars]
        ranks = [data['rank'] for _, data in sorted_chars]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(characteristics)))
        bars = ax.barh(range(len(characteristics)), scores, color=colors)
        
        ax.set_yticks(range(len(characteristics)))
        ax.set_yticklabels(characteristics)
        ax.set_xlabel('Importance Score')
        ax.set_title('Characteristic Ranking by Importance - Model V2')
        ax.invert_yaxis()
        
        for i, (bar, score, rank) in enumerate(zip(bars, scores, ranks)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'#{rank} ({score:.3f})', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'characteristic_ranking_v2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        print("Loading analysis results...")
        explainability_data, training_results = self.load_results()
        
        print("Creating feature importance plot...")
        self.create_feature_importance_plot(training_results)
        
        print("Creating characteristic importance stacked bar chart...")
        self.create_characteristic_importance_stacked_bar(explainability_data)
        
        print("Creating TD vs ASD comparison plot...")
        self.create_td_vs_asd_comparison_plot(explainability_data)
        
        print("Creating model performance plot...")
        self.create_model_performance_plot(training_results)
        
        print("Creating confusion matrix plot...")
        self.create_confusion_matrix_plot(training_results)
        
        print("Creating characteristic ranking plot...")
        self.create_characteristic_ranking_plot(explainability_data)
        
        print(f"All visualizations saved to: {self.viz_dir}")
        
        return self.viz_dir

def create_visualizations():
    visualizer = ModelVisualizer()
    viz_dir = visualizer.generate_all_visualizations()
    return viz_dir

if __name__ == "__main__":
    viz_dir = create_visualizations()