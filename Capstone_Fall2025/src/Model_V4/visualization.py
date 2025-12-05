import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

data_version = "Data_v4"
model_version = "V4"


# ---------------------------------------------------------
# NEW: Multi-format save utility
# ---------------------------------------------------------
def save_figure_multi_formats(fig, base_path, formats=("png", "pdf", "svg")):
    base_path = Path(base_path)
    for ext in formats:
        fig.savefig(base_path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


class ModelVisualizer:
    def __init__(self):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        self.results_dir = project_root / "Results" / model_version
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

        fig = plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Most Important Features - XGBoost Model {model_version}')
        plt.gca().invert_yaxis()

        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{importances[i]:.3f}', va='center', fontsize=8)

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f'feature_importance_{model_version}')
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
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=8)

        bars2 = ax2.bar(characteristics, feature_counts, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Number of Features')
        ax2.set_xlabel('Characteristics')
        ax2.set_title('Feature Count by Characteristic')
        ax2.tick_params(axis='x', rotation=45)

        for bar, count in zip(bars2, feature_counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{count}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f'characteristic_importance_{model_version}')
        plt.close()

    def create_td_vs_asd_comparison_plot(self, explainability_data):
        td_patterns, asd_patterns = explainability_data['td_vs_asd_patterns']
        characteristics = list(td_patterns.keys())

        td_vals = []
        asd_vals = []

        for char in characteristics:
            key = char.replace(" ", "_")
            td_vals.append(td_patterns[char].get(f"{key}_mentioned", 0))
            asd_vals.append(asd_patterns[char].get(f"{key}_mentioned", 0))

        x = np.arange(len(characteristics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(17, 8))
        ax.bar(x - width / 2, td_vals, width, label='TD', alpha=0.7, color='lightblue')
        ax.bar(x + width / 2, asd_vals, width, label='ASD', alpha=0.7, color='lightcoral')

        ax.set_xlabel('Characteristics')
        ax.set_ylabel('Avg Mention')
        ax.set_title(f'TD vs ASD Mention Patterns — Model {model_version}')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ").title() for c in characteristics], rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f'td_vs_asd_comparison_{model_version}')
        plt.close()

    def create_model_performance_plot(self, training_results):
        cv_scores = training_results['cv_scores']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.bar(range(len(cv_scores)), cv_scores, alpha=0.7, color='green')
        ax1.set_xlabel('CV Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Cross-Validation Scores - Model {model_version}')

        mean_score = training_results['cv_accuracy_mean']
        std_score = training_results['cv_accuracy_std']

        ax2.bar(['Mean CV Accuracy'], [mean_score], yerr=[std_score], capsize=10, alpha=0.7, color='blue')
        ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f'model_performance_{model_version}')
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

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['TD', 'ASD'], yticklabels=['TD', 'ASD'])
        plt.title(f'Confusion Matrix - Model {model_version}')

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f'confusion_matrix_{model_version}')
        plt.close()

    def create_feature_importance_by_target_plot(self, explainability_data):
        top_features = explainability_data['top_features'][:15]
        td_patterns, asd_patterns = explainability_data['td_vs_asd_patterns']

        feature_names = [f for f, _ in top_features]
        feature_importances = [imp for _, imp in top_features]

        td_vals = [td_patterns[f] for f in feature_names]
        asd_vals = [asd_patterns[f] for f in feature_names]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        ax1.bar(feature_names, feature_importances)
        ax1.set_title(f"Top Features — Importance (Model {model_version})")
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(feature_names, td_vals, label="TD", alpha=0.7)
        ax2.bar(feature_names, asd_vals, label="ASD", alpha=0.7)
        ax2.legend()
        ax2.set_title("Feature Values by Target Class")

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f'feature_importance_by_target_{model_version}')
        plt.close()


def create_visualizations():
    visualizer = ModelVisualizer()

    print("Loading results...")
    explainability_data, training_results = visualizer.load_results()

    print("Creating feature importance plot...")
    visualizer.create_feature_importance_plot(explainability_data)

    print("Creating feature importance by target plot...")
    visualizer.create_feature_importance_by_target_plot(explainability_data)

    print("Creating model performance plot...")
    visualizer.create_model_performance_plot(training_results)

    print("Creating confusion matrix plot...")
    visualizer.create_confusion_matrix_plot()

    print(f"All visualizations saved to: {visualizer.viz_dir}")


if __name__ == "__main__":
    create_visualizations()
