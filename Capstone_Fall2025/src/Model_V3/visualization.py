import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from explainability_analysis import ExplainabilityAnalyzer

# -------------------------------------------------------------------
# VERSION CONFIG
# -------------------------------------------------------------------
data_version = "Data_v3"
model_version = "V3"
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Multi-format save utility
# -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    def load_results(self):
        explainability_path = self.results_dir / f"explainability_analysis_{model_version}.json"
        training_results_path = self.results_dir / f"training_results_{model_version}.json"

        with open(explainability_path, "r") as f:
            explainability_data = json.load(f)

        with open(training_results_path, "r") as f:
            training_results = json.load(f)

        return explainability_data, training_results

    # -------------------------------------------------------------------
    def create_feature_importance_plot(self, explainability_data):
        top_features = explainability_data["top_features"][:20]
        features, importances = zip(*top_features)

        fig = plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Feature Importance")
        plt.title(f"Top 20 Most Important Features - XGBoost Model {model_version}")
        plt.gca().invert_yaxis()

        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{importances[i]:.3f}",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"feature_importance_{model_version}")
        plt.close()

    # -------------------------------------------------------------------
    def create_characteristic_importance_plot(self, explainability_data):
        char_analysis = explainability_data["characteristic_analysis"]

        characteristics = []
        importance_scores = []
        feature_counts = []

        for char, data in sorted(
            char_analysis.items(), key=lambda x: x[1]["total_importance"], reverse=True
        ):
            characteristics.append(char.replace("_", " ").title())
            importance_scores.append(data["total_importance"])
            feature_counts.append(data["feature_count"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Importance
        bars1 = ax1.bar(characteristics, importance_scores, color="skyblue", alpha=0.7)
        ax1.set_ylabel("Total Importance Score")
        ax1.set_title(f"Characteristic Importance Scores - Model {model_version}")
        ax1.tick_params(axis="x", rotation=45)

        for bar, score in zip(bars1, importance_scores):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Count
        bars2 = ax2.bar(characteristics, feature_counts, color="lightcoral", alpha=0.7)
        ax2.set_ylabel("Number of Features")
        ax2.set_xlabel("Characteristics")
        ax2.set_title("Feature Count by Characteristic")
        ax2.tick_params(axis="x", rotation=45)

        for bar, count in zip(bars2, feature_counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"characteristic_importance_{model_version}")
        plt.close()

    # -------------------------------------------------------------------
    def create_td_vs_asd_comparison_plot(self, explainability_data):
        td_patterns, asd_patterns = explainability_data["td_vs_asd_patterns"]
        characteristics = list(td_patterns.keys())

        td_mentioned = []
        asd_mentioned = []

        # Mention rates
        for char in characteristics:
            char_clean = char.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            td_mentioned.append(td_patterns[char].get(f"{char_clean}_mentioned", 0))
            asd_mentioned.append(asd_patterns[char].get(f"{char_clean}_mentioned", 0))

        # Load preprocessed data to calculate FSR + PE
        train_data_path = (
            self.results_dir.parent.parent
            / "data"
            / data_version
            / f"LLM_data_train_preprocessed_{model_version}.csv"
        )

        if train_data_path.exists():
            df = pd.read_csv(train_data_path)

            # FSR
            td_fsr = df[df["td_or_asd"] == 1]["FSR"].mean()
            asd_fsr = df[df["td_or_asd"] == 0]["FSR"].mean()
            fsr_min, fsr_max = df["FSR"].min(), df["FSR"].max()

            td_fsr_norm = (td_fsr - fsr_min) / (fsr_max - fsr_min) if fsr_max > fsr_min else 0
            asd_fsr_norm = (asd_fsr - fsr_min) / (fsr_max - fsr_min) if fsr_max > fsr_min else 0

            # PE
            td_pe = df[df["td_or_asd"] == 1]["avg_PE"].mean()
            asd_pe = df[df["td_or_asd"] == 0]["avg_PE"].mean()
            pe_min, pe_max = df["avg_PE"].min(), df["avg_PE"].max()

            td_pe_norm = (td_pe - pe_min) / (pe_max - pe_min) if pe_max > pe_min else 0
            asd_pe_norm = (asd_pe - pe_min) / (pe_max - pe_min) if pe_max > pe_min else 0

            # Add FSR + PE
            characteristics.extend(["FSR", "PE"])
            td_mentioned.extend([td_fsr_norm, td_pe_norm])
            asd_mentioned.extend([asd_fsr_norm, asd_pe_norm])

        x = np.arange(len(characteristics))
        width = 0.35

        fig = plt.figure(figsize=(17, 8))
        ax = plt.gca()

        bars1 = ax.bar(x - width / 2, td_mentioned, width, label="TD", alpha=0.7, color="lightblue")
        bars2 = ax.bar(x + width / 2, asd_mentioned, width, label="ASD", alpha=0.7, color="lightcoral")

        ax.set_xlabel("Characteristics")
        ax.set_ylabel("Average Mention Rate")
        ax.set_title(f"TD vs ASD: Characteristic Mention Patterns - Model {model_version}")

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ").title() for c in characteristics], rotation=45)
        ax.legend()

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"td_vs_asd_comparison_{model_version}")
        plt.close()

    # -------------------------------------------------------------------
    def create_model_performance_plot(self, training_results):
        cv_scores = training_results["cv_scores"]
        mean_score = training_results["cv_accuracy_mean"]
        std_score = training_results["cv_accuracy_std"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # CV fold scores
        ax1.bar(range(len(cv_scores)), cv_scores, alpha=0.7, color="green")
        ax1.set_xlabel("CV Fold")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"Cross-Validation Scores - Model {model_version}")

        # Overall performance
        ax2.bar(["Mean CV Accuracy"], [mean_score], yerr=[std_score], capsize=10, alpha=0.7)
        ax2.set_title("Overall Model Performance")

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"model_performance_{model_version}")
        plt.close()

    # -------------------------------------------------------------------
    def create_confusion_matrix_plot(self):
        test_results_path = self.results_dir / f"test_results_{model_version}.json"

        if not test_results_path.exists():
            print("No test_results found for confusion matrix plot.")
            return

        with open(test_results_path, "r") as f:
            test_results = json.load(f)

        cm = np.array(test_results["confusion_matrix"])

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["TD", "ASD"],
            yticklabels=["TD", "ASD"],
        )

        plt.title(f"Confusion Matrix - Model {model_version}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"confusion_matrix_{model_version}")
        plt.close()

    # -------------------------------------------------------------------
    def create_feature_importance_by_target_plot(self, explainability_data):
        top_features = explainability_data["top_features"][:15]
        td_patterns, asd_patterns = explainability_data["td_vs_asd_patterns"]

        feature_names = [feat[0] for feat in top_features]
        feature_importances = [feat[1] for feat in top_features]

        td_vals, asd_vals = [], []

        # Extract feature-specific TD/ASD values
        for feat in feature_names:
            td_val = 0
            asd_val = 0

            for char in td_patterns.keys():
                if feat.startswith(char.replace(" ", "_")):
                    td_val = td_patterns[char].get(feat, 0)
                    asd_val = asd_patterns[char].get(feat, 0)
                    break

            td_vals.append(td_val)
            asd_vals.append(asd_val)

        x = np.arange(len(feature_names))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Feature importance
        ax1.bar(x, feature_importances, color="steelblue", alpha=0.8)
        ax1.set_title(f"Top Features: Importance and Group Comparison - Model {model_version}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names, rotation=45, ha="right")

        # TD vs ASD values
        ax2.bar(x - width / 2, td_vals, width, label="TD", alpha=0.7)
        ax2.bar(x + width / 2, asd_vals, width, label="ASD", alpha=0.7)
        ax2.set_title("Feature Values by Class (TD vs ASD)")
        ax2.legend()

        plt.tight_layout()
        save_figure_multi_formats(fig, self.viz_dir / f"feature_importance_by_target_{model_version}")
        plt.close()

    # -------------------------------------------------------------------
    def generate_all(self):
        print("Loading results...")
        explainability_data, training_results = self.load_results()

        print("Feature importance plot...")
        self.create_feature_importance_plot(explainability_data)

        print("Characteristic importance plot...")
        self.create_characteristic_importance_plot(explainability_data)

        print("TD vs ASD comparison plot...")
        self.create_td_vs_asd_comparison_plot(explainability_data)

        print("Feature importance by target plot...")
        self.create_feature_importance_by_target_plot(explainability_data)

        print("Model performance plot...")
        self.create_model_performance_plot(training_results)

        print("Confusion matrix plot...")
        self.create_confusion_matrix_plot()

        print(f"All visualizations saved to: {self.viz_dir}")
        return self.viz_dir


def create_visualizations():
    visualizer = ModelVisualizer()
    return visualizer.generate_all()


if __name__ == "__main__":
    create_visualizations()
