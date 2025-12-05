import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from xgboost_model import XGBoostClassifier
from explainability_analysis import ExplainabilityAnalyzer

data_version = "OverlapTest"
model_version = "V1"

class ModelVisualizer:
    def __init__(self):
        current_dir = Path(__file__).parent
        experiment_root = current_dir.parent
        self.results_dir = experiment_root / "Results" / data_version /model_version
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_results(self):
        explainability_path = self.results_dir / f'explainability_analysis_{model_version}.json'
        training_results_path = self.results_dir / f'training_results_{model_version}.json'
        
        with open(explainability_path, 'r') as f:
            explainability_data = json.load(f)
        
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        
        return explainability_data, training_results
    
    def create_feature_importance_plot(self, explainability_data):
        top_features = explainability_data['top_features'][:20]
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Most Important Features - XGBoost Model {model_version}')
        plt.gca().invert_yaxis()
        
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importances[i]:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'feature_importance_{model_version}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_characteristic_importance_plot(self, explainability_data):
        char_analysis = explainability_data['characteristic_analysis']
        
        characteristics = []
        importance_scores = []
        feature_counts = []
        
        for char, data in sorted(char_analysis.items(), key=lambda x: x[1]['total_importance'], reverse=True):
            characteristics.append(char.replace('_', ' ').title())
            importance_scores.append(data['total_importance'])
            feature_counts.append(data['feature_count'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        bars1 = ax1.bar(characteristics, importance_scores, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Total Importance Score')
        ax1.set_title(f'Characteristic Importance Scores - Model {model_version}')
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
        plt.savefig(self.viz_dir / f'characteristic_importance_{model_version}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_td_vs_asd_comparison_plot(self, explainability_data):
        td_patterns, asd_patterns = explainability_data['td_vs_asd_patterns']
        
        # Get characteristics and add FSR and PE
        characteristics = list(td_patterns.keys())
        
        td_mentioned = []
        asd_mentioned = []
        
        # Process characteristic features
        for char in characteristics:
            char_clean = char.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            td_char_data = td_patterns[char]
            asd_char_data = asd_patterns[char]
            
            td_mentioned.append(td_char_data.get(f'{char_clean}_mentioned', 0))
            asd_mentioned.append(asd_char_data.get(f'{char_clean}_mentioned', 0))
        
        # Load preprocessed data to calculate FSR and PE averages by group
        train_data_path = self.results_dir.parent.parent / "data" / data_version / f"LLM_data_train_preprocessed_{model_version}.csv"
        if train_data_path.exists():
            train_df = pd.read_csv(train_data_path)
            
            # Calculate FSR averages
            td_fsr_avg = train_df[train_df['td_or_asd'] == 1]['FSR'].mean()
            asd_fsr_avg = train_df[train_df['td_or_asd'] == 0]['FSR'].mean()
            
            # Calculate PE averages
            td_pe_avg = train_df[train_df['td_or_asd'] == 1]['avg_PE'].mean()
            asd_pe_avg = train_df[train_df['td_or_asd'] == 0]['avg_PE'].mean()
            
            # Normalize FSR and PE to match the scale of mention rates (0-1)
            # FSR normalization - scale to 0-1 range based on data range
            fsr_min = train_df['FSR'].min()
            fsr_max = train_df['FSR'].max()
            td_fsr_norm = (td_fsr_avg - fsr_min) / (fsr_max - fsr_min) if fsr_max > fsr_min else 0
            asd_fsr_norm = (asd_fsr_avg - fsr_min) / (fsr_max - fsr_min) if fsr_max > fsr_min else 0
            
            # PE normalization - scale to 0-1 range based on data range  
            pe_min = train_df['avg_PE'].min()
            pe_max = train_df['avg_PE'].max()
            td_pe_norm = (td_pe_avg - pe_min) / (pe_max - pe_min) if pe_max > pe_min else 0
            asd_pe_norm = (asd_pe_avg - pe_min) / (pe_max - pe_min) if pe_max > pe_min else 0
            
            # Add FSR and PE to the data
            characteristics.extend(['FSR', 'PE'])
            td_mentioned.extend([td_fsr_norm, td_pe_norm])
            asd_mentioned.extend([asd_fsr_norm, asd_pe_norm])
        
        x = np.arange(len(characteristics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(17, 8))
        bars1 = ax.bar(x - width/2, td_mentioned, width, label='TD', alpha=0.7, color='lightblue')
        bars2 = ax.bar(x + width/2, asd_mentioned, width, label='ASD', alpha=0.7, color='lightcoral')
        
        ax.set_xlabel('Characteristics')
        ax.set_ylabel('Average Mention Rate')
        ax.set_title(f'TD vs ASD: Characteristic Mention Patterns - Model {model_version}')
        ax.set_xticks(x)
        ax.set_xticklabels([char.replace('_', ' ').title() for char in characteristics], rotation=45, ha='right')
        ax.legend()
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'td_vs_asd_comparison_{model_version}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_performance_plot(self, training_results):
        cv_scores = training_results['cv_scores']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.bar(range(len(cv_scores)), cv_scores, alpha=0.7, color='green')
        ax1.set_xlabel('CV Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Cross-Validation Scores - Model {model_version}')
        ax1.set_ylim([min(cv_scores) - 0.01, max(cv_scores) + 0.01])
        
        for i, score in enumerate(cv_scores):
            ax1.text(i, score + 0.002, f'{score:.3f}', ha='center', va='bottom')
        
        mean_score = training_results['cv_accuracy_mean']
        std_score = training_results['cv_accuracy_std']
        
        ax2.bar(['Mean CV Accuracy'], [mean_score], yerr=[std_score], 
                capsize=10, alpha=0.7, color='blue')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Overall Model Performance')
        ax2.set_ylim([mean_score - 3*std_score, mean_score + 3*std_score])
        
        ax2.text(0, mean_score + std_score + 0.005, 
                f'{mean_score:.3f} ± {std_score:.3f}', 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'model_performance_{model_version}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrix_plot(self, test_results_path=None):
        if test_results_path is None:
            test_results_path = self.results_dir / f'test_results_{model_version}.json'
        
        if not test_results_path.exists():
            print("Test results not found. Skipping confusion matrix plot.")
            return
        
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
        
        cm = np.array(test_results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['TD', 'ASD'], yticklabels=['TD', 'ASD'])
        plt.title(f'Confusion Matrix - Model {model_version}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'confusion_matrix_{model_version}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_by_target_plot(self, explainability_data):
        """Create stacked bar chart comparing feature importance by target class (TD vs ASD)"""
        
        # Get top features and TD vs ASD patterns
        top_features = explainability_data['top_features'][:15]  # Top 15 features
        td_patterns, asd_patterns = explainability_data['td_vs_asd_patterns']
        
        feature_names = [feat[0] for feat in top_features]
        feature_importances = [feat[1] for feat in top_features]
        
        # Calculate average values for TD and ASD groups for each feature
        td_values = []
        asd_values = []
        
        for feature_name in feature_names:
            # Look for the feature in TD and ASD patterns
            td_val = 0
            asd_val = 0
            
            # Search through characteristics for matching features
            for char in td_patterns.keys():
                char_clean = char.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
                if feature_name.startswith(char_clean):
                    td_val = td_patterns[char].get(feature_name, 0)
                    asd_val = asd_patterns[char].get(feature_name, 0)
                    break
            
            td_values.append(td_val)
            asd_values.append(asd_val)
        
        # Create the plot
        x = np.arange(len(feature_names))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Top plot: Feature importance
        bars1 = ax1.bar(x, feature_importances, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Feature Importance')
        ax1.set_title(f'Top Features: Importance and Target Class Comparison - Model {model_version}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, importance in zip(bars1, feature_importances):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Bottom plot: Stacked comparison by target class
        bars2 = ax2.bar(x - width/2, td_values, width, label='TD', alpha=0.7, color='lightblue')
        bars3 = ax2.bar(x + width/2, asd_values, width, label='ASD', alpha=0.7, color='lightcoral')
        
        ax2.set_ylabel('Average Feature Value')
        ax2.set_xlabel('Features')
        ax2.set_title('Feature Values by Target Class (TD vs ASD)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        ax2.legend()
        
        # Add value labels
        for bar, val in zip(bars2, td_values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars3, asd_values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'feature_importance_by_target_{model_version}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance by target plot saved to: {self.viz_dir / f'feature_importance_by_target_{model_version}.png'}")

def create_visualizations():
    visualizer = ModelVisualizer()
    
    print("Loading results...")
    explainability_data, training_results = visualizer.load_results()
    
    print("Creating feature importance plot...")
    visualizer.create_feature_importance_plot(explainability_data)
    
    print("Creating characteristic importance plot...")
    visualizer.create_characteristic_importance_plot(explainability_data)
    
    print("Creating TD vs ASD comparison plot...")
    visualizer.create_td_vs_asd_comparison_plot(explainability_data)
    
    print("Creating feature importance by target plot...")
    visualizer.create_feature_importance_by_target_plot(explainability_data)
    
    print("Creating model performance plot...")
    visualizer.create_model_performance_plot(training_results)
    
    print("Creating confusion matrix plot...")
    visualizer.create_confusion_matrix_plot()
    
    print(f"All visualizations saved to: {visualizer.viz_dir}")

if __name__ == "__main__":
    create_visualizations()