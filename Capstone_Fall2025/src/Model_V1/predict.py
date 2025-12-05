import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from xgboost_model import XGBoostClassifier
from data_preprocessor import preprocess_prediction_data
from explainability_analysis import run_explainability_analysis
from visualization import create_visualizations, save_figure_multi_formats


data_version = "Data_v1"
model_version = "V1"


class ModelPredictor:
    def __init__(self):
        self.classifier = XGBoostClassifier()

        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent

        self.results_dir = project_root / "Results" / model_version
        self.predictions_dir = self.results_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        self.classifier.load_model()
        print("Model loaded successfully.")

    def preprocess_new_data(self, df, is_test_data=False):
        return preprocess_prediction_data(df, is_test_data=is_test_data)

    def predict_batch(self, df, is_test_data=False):
        processed_df = self.preprocess_new_data(df, is_test_data=is_test_data)
        X, _ = self.classifier.prepare_features(processed_df)
        predictions, probabilities = self.classifier.predict(X)

        results_df = processed_df.copy()
        results_df["predicted_td_or_asd"] = predictions
        results_df["prediction_probability_class_0"] = probabilities[:, 0]
        results_df["prediction_probability_class_1"] = probabilities[:, 1]
        results_df["prediction_confidence"] = np.max(probabilities, axis=1)

        return results_df

    def predict_single_text(self, text, subject_id="unknown"):
        single_df = pd.DataFrame({
            "sub": [subject_id],
            "profile": ["unknown"],
            "subject": [subject_id],
            "td_or_asd": [0],
            "SRS.Raw": [0],
            "FSR": [0],
            "BIS": [0],
            "avg_PE": [0],
            "free_response": [text],
            "LPA_Profile_grand_mean": [0],
            "LPA_Profile_ASD_only": [0]
        })

        result = self.predict_batch(single_df)
        return result.iloc[0]

    def create_prediction_visualization(self, results_df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Prediction distribution pie
        prediction_counts = results_df["predicted_td_or_asd"].value_counts()
        axes[0, 0].pie(
            prediction_counts.values,
            labels=["TD", "ASD"],
            autopct="%1.1f%%",
            startangle=90
        )
        axes[0, 0].set_title("Prediction Distribution")

        # 2. Confidence histogram
        axes[0, 1].hist(
            results_df["prediction_confidence"],
            bins=20,
            alpha=0.7,
            color="skyblue",
            edgecolor="black"
        )
        axes[0, 1].set_title("Prediction Confidence Distribution")
        axes[0, 1].set_xlabel("Confidence")

        # 3. Mean confidence by class
        conf_by_class = results_df.groupby("predicted_td_or_asd")["prediction_confidence"].mean()
        axes[1, 0].bar(["TD", "ASD"], conf_by_class.values,
                       color=["lightblue", "lightcoral"], alpha=0.7)
        axes[1, 0].set_title("Average Confidence by Class")

        for i, v in enumerate(conf_by_class):
            axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha="center")

        # 4. Certainty scatter
        prob_diff = np.abs(
            results_df["prediction_probability_class_1"]
            - results_df["prediction_probability_class_0"]
        )
        axes[1, 1].scatter(
            results_df.index, prob_diff, alpha=0.6,
            c=results_df["predicted_td_or_asd"], cmap="coolwarm"
        )
        axes[1, 1].axhline(0.5, color="red", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Prediction Certainty by Sample")
        axes[1, 1].set_ylabel("|P(ASD) - P(TD)|")

        plt.tight_layout()

        if save_path:
            base_path = Path(str(save_path).replace(".png", ""))
            save_figure_multi_formats(fig, base_path)
            print(f"Saved prediction visualization → {base_path}.[png/pdf/svg]")

        plt.close()

    def create_feature_contribution_plot(self, sample_result, top_n=15):
        excluded_cols = {
            "sub", "profile", "subject", "td_or_asd", "free_response",
            "predicted_td_or_asd", "prediction_probability_class_0",
            "prediction_probability_class_1", "prediction_confidence"
        }

        feature_cols = [c for c in sample_result.index if c not in excluded_cols]
        feature_values = sample_result[feature_cols]
        feature_importance = self.classifier.feature_importance

        contributions = [
            (feat, feature_values[feat] * feature_importance.get(feat, 0), feature_values[feat])
            for feat in feature_cols
        ]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_contribs = contributions[:top_n]

        features, contribs, values = zip(*top_contribs)

        plt.figure(figsize=(12, 8))
        colors = ["red" if c < 0 else "blue" for c in contribs]
        bars = plt.barh(range(len(features)), contribs, color=colors, alpha=0.7)

        plt.yticks(range(len(features)), features)
        plt.title(f"Top {top_n} Feature Contributions - Model {model_version}")
        plt.axvline(0, color="black", linestyle="-", alpha=0.4)

        for bar, contrib, value in zip(bars, contribs, values):
            plt.text(bar.get_width() + (0.001 if contrib >= 0 else -0.001),
                     bar.get_y() + bar.get_height() / 2,
                     f"{contrib:.3f} (val: {value:.2f})",
                     va="center",
                     ha="left" if contrib >= 0 else "right",
                     fontsize=8)

        plt.tight_layout()

        base_path = self.predictions_dir / f"sample_feature_contributions_{model_version}"
        save_figure_multi_formats(plt.gcf(), base_path)
        plt.close()

        return f"{base_path}.png"

    def generate_prediction_report(self, results_df):
        return {
            "total_samples": len(results_df),
            "td_predictions": int((results_df["predicted_td_or_asd"] == 0).sum()),
            "asd_predictions": int((results_df["predicted_td_or_asd"] == 1).sum()),
            "average_confidence": float(results_df["prediction_confidence"].mean()),
            "high_confidence_predictions": int((results_df["prediction_confidence"] > 0.8).sum()),
            "low_confidence_predictions": int((results_df["prediction_confidence"] < 0.6).sum()),
            "prediction_summary": {
                "td_percentage": float((results_df["predicted_td_or_asd"] == 0).mean() * 100),
                "asd_percentage": float((results_df["predicted_td_or_asd"] == 1).mean() * 100)
            }
        }

    def save_predictions(self, results_df, filename=None):
        if filename is None:
            filename = f"predictions_{model_version}.csv"

        output_path = self.predictions_dir / filename
        results_df.to_csv(output_path, index=False)
        print(f"Saved predictions → {output_path}")
        return output_path


def predict_from_file(data_path):
    print("=" * 60)
    print(f"STARTING MODEL {model_version} TEST PIPELINE")
    print("=" * 60)

    predictor = ModelPredictor()
    predictor.load_model()

    df = pd.read_csv(data_path)
    print(f"Loaded test data: {df.shape}")

    results_df = predictor.predict_batch(df, is_test_data=True)

    y_true = results_df["td_or_asd"]
    y_pred = results_df["predicted_td_or_asd"]
    y_proba = results_df["prediction_probability_class_1"]

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)

    roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else None

    test_metrics = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "roc_auc": roc_auc
    }

    analyzer, explainability_report, _ = run_explainability_analysis()

    viz_dir = create_visualizations()

    viz_path = predictor.predictions_dir / f"test_prediction_analysis_{model_version}.png"
    predictor.create_prediction_visualization(results_df, viz_path)

    sample_idx = results_df["prediction_confidence"].idxmax()
    sample_result = results_df.loc[sample_idx]
    predictor.create_feature_contribution_plot(sample_result)

    predictions_path = predictor.save_predictions(results_df)

    summary = predictor.generate_prediction_report(results_df)

    comprehensive_results = {
        "test_metrics": test_metrics,
        "prediction_summary": summary,
        "explainability_summary": {
            "top_characteristics": sorted(
                explainability_report["characteristic_summary"].items(),
                key=lambda x: x[1]["importance_score"],
                reverse=True
            )[:10]
        }
    }

    results_path = predictor.results_dir / f"comprehensive_test_results_{model_version}.json"
    with open(results_path, "w") as f:
        json.dump(comprehensive_results, f, indent=2)

    print("=" * 60)
    print(f"MODEL {model_version} TEST PIPELINE COMPLETE")
    print("=" * 60)

    return results_df, comprehensive_results


def predict_single_sample(text, subject_id="sample"):
    predictor = ModelPredictor()
    predictor.load_model()

    result = predictor.predict_single_text(text, subject_id)

    print(f"\nPrediction for {subject_id}:")
    print(f"Predicted class: {'ASD' if result['predicted_td_or_asd'] == 1 else 'TD'}")
    print(f"Confidence: {result['prediction_confidence']:.3f}")

    return result


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    test_data_path = project_root / "data" / data_version / "LLM_data_test.csv"

    if test_data_path.exists():
        predict_from_file(test_data_path)
    else:
        print(f"Test data missing: {test_data_path}")
