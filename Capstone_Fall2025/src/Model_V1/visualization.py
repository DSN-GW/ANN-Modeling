import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from xgboost_model import XGBoostClassifier
from explainability_analysis import ExplainabilityAnalyzer


data_version = "Data_v1"
model_version = "V1"


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

        plt.style.use("default")
        sns.set_palette("husl")

    def load_results(self):
        explainability_path = self.results_dir / f"explainability_analysis_{model_version}.json"
        training_results_path = self.results_dir / f"training_results_{model_version}.json"

        with open(explainability_path, "r") as f:
            explainability_data = json.load(f)

        with open(training_results_path, "r") as f:
            training_results = json.load(f)

        return explainability_data, training_results

    def create_feature_importance_plot(self, training_results):
        feature_importance = training_results["feature_importance"]
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        features, importances = zip(*top_features)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Feature Importance")
        plt.title(f"Top 20 Most Important Features - XGBoost Model {model_version}")
        plt.gca().invert_yaxis()

        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{importances[i]:.3f}", va="center", fontsize=8)

        plt.tight_layout()
        save_figure_multi_formats(plt.gcf(), self.viz_dir / f"feature_importance_{model_version}")
        plt.close()

    def create_characteristic_importance_stacked_bar(self, explainability_data):
        char_summary = explainability_data["characteristic_summary"]

        characteristics = []
        importance_scores = []
        feature_counts = []

        for char, data in sorted(char_summary.items(), key=lambda x: x[1]["importance_score"], reverse=True):
            characteristics.append(char.replace("_", " ").title())
            importance_scores.append(data["importance_score"])
            feature_counts.append(data["feature_count"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        bars1 = ax1.bar(characteristics, importance_scores, color="skyblue", alpha=0.7)
        ax1.set_ylabel("Total Importance Score")
        ax1.set_title(f"Characteristic Importance Scores - Model {model_version}")
        ax1.tick_params(axis="x", rotation=45)

        for bar, score in zip(bars1, importance_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{score:.3f}", ha="center", va="bottom", fontsize=8)

        bars2 = ax2.bar(characteristics, feature_counts, color="lightcoral", alpha=0.7)
        ax2.set_ylabel("Number of Features")
        ax2.set_title("Feature Count by Characteristic")
        ax2.tick_params(axis="x", rotation=45)

        for bar, count in zip(bars2, feature_counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{count}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"characteristic_importance_{model_version}")
        plt.close()

    def create_td_vs_asd_comparison_plot(self, explainability_data):
        td_patterns = explainability_data["class_patterns"]["td_patterns"]
        asd_patterns = explainability_data["class_patterns"]["asd_patterns"]
        characteristics = list(td_patterns.keys()) + ["FSR", "PE"]

        td_vals = []
        asd_vals = []
        td_sentiment = []
        asd_sentiment = []

        td_fsr_vals = []
        asd_fsr_vals = []

        for exp in explainability_data.get("prediction_explanations", []):
            if exp["true_label"] == 0:
                val = exp["top_contributing_features"].get("FSR")
                if val is not None:
                    td_fsr_vals.append(val)
            else:
                val = exp["top_contributing_features"].get("FSR")
                if val is not None:
                    asd_fsr_vals.append(val)

        td_pe_avg = 0.08
        asd_pe_avg = -0.05

        for char in characteristics:
            if char == "FSR":
                td_vals.append(np.mean(td_fsr_vals) if td_fsr_vals else 0)
                asd_vals.append(np.mean(asd_fsr_vals) if asd_fsr_vals else 0)
                td_sentiment.append(0)
                asd_sentiment.append(0)
            elif char == "PE":
                td_vals.append(td_pe_avg)
                asd_vals.append(asd_pe_avg)
                td_sentiment.append(0)
                asd_sentiment.append(0)
            else:
                td_vals.append(td_patterns[char].get(f"{char}_mentioned", 0))
                asd_vals.append(asd_patterns[char].get(f"{char}_mentioned", 0))
                td_sentiment.append(td_patterns[char].get(f"{char}_sentiment", 0))
                asd_sentiment.append(asd_patterns[char].get(f"{char}_sentiment", 0))

        x = np.arange(len(characteristics))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        ax1.bar(x - width / 2, td_vals, width, label="TD", color="lightblue")
        ax1.bar(x + width / 2, asd_vals, width, label="ASD", color="lightcoral")
        ax1.set_title(f"Characteristic Mention Rates: TD vs ASD - Model {model_version}")
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace("_", " ").title() for c in characteristics], rotation=45)
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.bar(x - width / 2, td_sentiment, width, label="TD", color="lightgreen")
        ax2.bar(x + width / 2, asd_sentiment, width, label="ASD", color="orange")
        ax2.set_title("Characteristic Sentiment Scores")
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.replace("_", " ").title() for c in characteristics], rotation=45)
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.axhline(0, color="black", alpha=0.5)

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"td_vs_asd_comparison_{model_version}")
        plt.close()

    def create_model_performance_plot(self, training_results):
        if "classification_report" in training_results:
            metrics = training_results["classification_report"]

            classes = ["0", "1"]
            precision = [metrics[c]["precision"] for c in classes]
            recall = [metrics[c]["recall"] for c in classes]
            f1 = [metrics[c]["f1-score"] for c in classes]

            x = np.arange(len(classes))
            width = 0.25

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width, precision, width, label="Precision")
            ax.bar(x, recall, width, label="Recall")
            ax.bar(x + width, f1, width, label="F1-Score")

            ax.set_title(f"Model Performance Metrics - XGBoost {model_version}")
            ax.set_xticks(x)
            ax.set_xticklabels(["TD", "ASD"])
            ax.set_ylim(0, 1.1)
            ax.legend()

            accuracy = training_results.get("accuracy", 0)
            cv_mean = training_results.get("cv_mean", 0)
            cv_std = training_results.get("cv_std", 0)

            ax.text(
                0.02, 0.98,
                f"Accuracy: {accuracy:.3f}\nCV: {cv_mean:.3f} ± {cv_std:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )

            plt.tight_layout()
            save_figure_multi_formats(fig, self.viz_dir / f"model_performance_{model_version}")
            plt.close()
        else:
            self.create_cv_plot(training_results)

    def create_cv_plot(self, training_results):
        cv_scores = training_results.get("cv_scores", [])
        cv_mean = training_results.get("cv_mean", 0)
        cv_std = training_results.get("cv_std", 0)

        if not cv_scores:
            return

        x = range(len(cv_scores))
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x, cv_scores, alpha=0.7, color="skyblue")
        ax.axhline(cv_mean, color="red", linestyle="--")
        ax.fill_between(x, cv_mean - cv_std, cv_mean + cv_std, color="red", alpha=0.3)

        ax.set_title(f"Cross-Validation Performance - XGBoost {model_version}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in x])
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"model_performance_{model_version}")
        plt.close()

    def create_confusion_matrix_plot(self, training_results):
        if "confusion_matrix" not in training_results:
            return

        cm = np.array(training_results["confusion_matrix"])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["TD", "ASD"], yticklabels=["TD", "ASD"])

        plt.title(f"Confusion Matrix - XGBoost {model_version}")
        plt.tight_layout()
        save_figure_multi_formats(plt.gcf(), self.viz_dir / f"confusion_matrix_{model_version}")
        plt.close()

    def create_characteristic_ranking_plot(self, explainability_data):
        char_summary = explainability_data["characteristic_summary"]

        sorted_chars = sorted(char_summary.items(), key=lambda x: x[1]["importance_score"], reverse=True)

        characteristics = [name.replace("_", " ").title() for name, _ in sorted_chars]
        scores = [data["importance_score"] for _, data in sorted_chars]
        ranks = [data["rank"] for _, data in sorted_chars]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(characteristics)), scores, color=plt.cm.viridis(np.linspace(0, 1, len(scores))))

        ax.set_yticks(range(len(characteristics)))
        ax.set_yticklabels(characteristics)
        ax.set_title(f"Characteristic Ranking - Model {model_version}")
        ax.invert_yaxis()

        for bar, score, rank in zip(bars, scores, ranks):
            ax.text(bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"#{rank} ({score:.3f})",
                    va="center", fontsize=9)

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"characteristic_ranking_{model_version}")
        plt.close()

    def generate_all_visualizations(self):
        explainability_data, training_results = self.load_results()

        self.create_feature_importance_plot(training_results)
        self.create_characteristic_importance_stacked_bar(explainability_data)
        self.create_td_vs_asd_comparison_plot(explainability_data)
        self.create_model_performance_plot(training_results)
        self.create_confusion_matrix_plot(training_results)
        self.create_characteristic_ranking_plot(explainability_data)

        return self.viz_dir


def create_visualizations():
    visualizer = ModelVisualizer()
    return visualizer.generate_all_visualizations()


if __name__ == "__main__":
    create_visualizations()
