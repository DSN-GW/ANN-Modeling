import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost_model import XGBoostClassifier
from data_preprocessor import preprocess_prediction_data
from explainability_analysis import run_explainability_analysis
from visualization import create_visualizations   # ← ADDED

data_version = "Data_v6"
model_version = "V6"


# -------------------------------------------------------
# MULTI-FORMAT SAVE FUNCTION
# -------------------------------------------------------
def save_figure_multi_formats(fig, base_path, formats=("png", "pdf", "svg")):
    base_path = Path(base_path)
    for ext in formats:
        fig.savefig(base_path.with_suffix(f".{ext}"),
                    dpi=300, bbox_inches="tight")


# -------------------------------------------------------
# MODEL PREDICTOR CLASS
# -------------------------------------------------------
class ModelPredictor:
    def __init__(self):
        self.classifier = XGBoostClassifier()
        self.setup_results_directory()

    def setup_results_directory(self):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        self.results_dir = project_root / "Results" / model_version
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        self.classifier.load_model()

    def preprocess_new_data(self, df, is_test_data=False):
        return preprocess_prediction_data(df, is_test_data=is_test_data)

    def predict_batch(self, df, is_test_data=False):
        processed_df = self.preprocess_new_data(df, is_test_data)
        X, _ = self.classifier.prepare_features(processed_df)
        predictions, probabilities = self.classifier.predict(X)

        results_df = processed_df.copy()
        results_df["predicted_td_or_asd"] = predictions
        results_df["prediction_probability"] = probabilities[:, 1]

        return results_df

    # -------------------------------------------------------
    # MULTIFORMAT PREDICTION VISUALIZATION
    # -------------------------------------------------------
    def create_prediction_visualization(self, results_df):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        if "td_or_asd" in results_df.columns:
            cm = confusion_matrix(results_df["td_or_asd"],
                                  results_df["predicted_td_or_asd"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
            axes[0, 0].set_title("Confusion Matrix")

        # Probability Distribution
        results_df["prediction_probability"].hist(bins=20, ax=axes[0, 1])
        axes[0, 1].set_title("Prediction Probability Distribution")

        # Prediction Count Plot
        prediction_counts = results_df["predicted_td_or_asd"].value_counts()
        axes[1, 0].bar(prediction_counts.index, prediction_counts.values)
        axes[1, 0].set_title("Prediction Distribution")

        # Accuracy
        if "td_or_asd" in results_df.columns:
            accuracy = accuracy_score(results_df["td_or_asd"],
                                      results_df["predicted_td_or_asd"])
            axes[1, 1].text(0.5, 0.5, f"Accuracy: {accuracy:.4f}",
                            ha="center", va="center", fontsize=16)
            axes[1, 1].set_title("Model Performance")

        plt.tight_layout()

        base_path = (self.results_dir /
                     "predictions" /
                     f"test_prediction_analysis_{model_version}")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure_multi_formats(fig, base_path)

        plt.close()
        print(f"Prediction visualization saved: {base_path}.[png/pdf/svg]")

    # -------------------------------------------------------
    # SAVE CSV
    # -------------------------------------------------------
    def save_predictions(self, results_df, filename=f"test_predictions_{model_version}.csv"):
        predictions_dir = self.results_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        save_path = predictions_dir / filename
        results_df.to_csv(save_path, index=False)
        print(f"Predictions saved to: {save_path}")


# -------------------------------------------------------
# MAIN PREDICTION LOGIC
# -------------------------------------------------------
def predict_from_file(data_path):
    predictor = ModelPredictor()

    print("Loading trained model...")
    predictor.load_model()

    print(f"Loading test data from: {data_path}")
    test_df = pd.read_csv(data_path)
    print(f"Test data shape: {test_df.shape}")

    print("Making predictions...")
    results_df = predictor.predict_batch(test_df, is_test_data=True)

    # -------------------------------------------------------
    # FULL MODEL VISUALIZATION + EXPLAINABILITY
    # -------------------------------------------------------
    print("Running explainability analysis...")
    run_explainability_analysis()     # SHAP + LIME, all saved in Results/V6/explainability/

    print("Creating full model visualizations...")
    create_visualizations()           # feature importance, performance, etc.

    # -------------------------------------------------------
    # SAVE PREDICTIONS + PREDICTION VISUALS
    # -------------------------------------------------------
    print("Saving predictions...")
    predictor.save_predictions(results_df)

    # Metrics
    if "td_or_asd" in results_df.columns:
        accuracy = accuracy_score(results_df["td_or_asd"],
                                  results_df["predicted_td_or_asd"])
        print(f"\nTest Accuracy: {accuracy:.4f}")

        test_results = {
            "accuracy": accuracy,
            "classification_report": classification_report(
                results_df["td_or_asd"],
                results_df["predicted_td_or_asd"],
                output_dict=True),
            "confusion_matrix": confusion_matrix(
                results_df["td_or_asd"],
                results_df["predicted_td_or_asd"]).tolist(),
        }

        results_path = predictor.results_dir / f"test_results_{model_version}.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)

        print(f"Test results saved to: {results_path}")

        predictor.create_prediction_visualization(results_df)

    return results_df


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    test_data_path = project_root / "data" / data_version / f"LLM_data_test_{model_version}.csv"

    if not test_data_path.exists():
        print(f"Test data not found at {test_data_path}")
        print("Run train.py first.")
        return

    results_df = predict_from_file(test_data_path)

    print("\n" + "=" * 60)
    print(f"MODEL {model_version} PREDICTION PIPELINE COMPLETED")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
