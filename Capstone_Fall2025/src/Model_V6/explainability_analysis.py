import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from xgboost_model import XGBoostClassifier

data_version = "Data_v6"
model_version = "V6"


class ExplainabilityAnalyzer:
    def __init__(self):
        self.classifier = XGBoostClassifier()
        self.explainer = None
        self.shap_values = None
        self.characteristics = self.load_characteristics()

    # ---------------------------------------------------------
    # Load characteristic list
    # ---------------------------------------------------------
    def load_characteristics(self):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        char_path = project_root / "data" / data_version / "charactristic.txt"

        with open(char_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    # ---------------------------------------------------------
    # Load model + clean training data
    # ---------------------------------------------------------
    def load_model_and_data(self):
        self.classifier.load_model()
        df = self.classifier.load_data()
        X, y = self.classifier.prepare_features(df)
        return df, X, y

    # ---------------------------------------------------------
    # SHAP explainer
    # ---------------------------------------------------------
    def initialize_explainer(self, X_sample):
        X_scaled = self.classifier.scaler.transform(X_sample)
        self.explainer = shap.TreeExplainer(self.classifier.model)
        self.shap_values = self.explainer.shap_values(X_scaled)
        return self.shap_values

    # ---------------------------------------------------------
    # Characteristic-level importance aggregation
    # ---------------------------------------------------------
    def analyze_characteristic_contributions(self, X, y):
        contributions = {}

        for char in self.characteristics:
            prefix = char.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            char_feats = [c for c in X.columns if c.startswith(prefix)]

            if not char_feats:
                continue

            total = sum(
                self.classifier.feature_importance.get(f, 0)
                for f in char_feats
            )

            contributions[char] = {
                "total_importance": float(total),
                "features": char_feats,
                "feature_count": len(char_feats),
            }

        return contributions

    # ---------------------------------------------------------
    # TD vs ASD — PER FEATURE AVERAGES (used for heatmaps/plots)
    # ---------------------------------------------------------
    def analyze_td_vs_asd_feature_means(self, X, y):
        td_avg = X[y == 0].mean().to_dict()
        asd_avg = X[y == 1].mean().to_dict()
        return td_avg, asd_avg

    # ---------------------------------------------------------
    # TD vs ASD — PER CHARACTERISTIC (for grouped plots)
    # ---------------------------------------------------------
    def analyze_td_vs_asd_characteristics(self, X, y):
        td_patterns = {}
        asd_patterns = {}

        for char in self.characteristics:
            prefix = char.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            char_feats = [c for c in X.columns if c.startswith(prefix)]
            if not char_feats:
                continue

            td_patterns[char] = X[y == 0][char_feats].mean().to_dict()
            asd_patterns[char] = X[y == 1][char_feats].mean().to_dict()

        return td_patterns, asd_patterns

    # ---------------------------------------------------------
    # Top XGBoost features
    # ---------------------------------------------------------
    def get_top_discriminative_features(self, n_features=20):
        fi = self.classifier.feature_importance
        if fi is None:
            return []
        return sorted(fi.items(), key=lambda x: x[1], reverse=True)[:n_features]

    # ---------------------------------------------------------
    # Text pattern analysis per class
    # ---------------------------------------------------------
    def analyze_text_patterns_by_class(self, df):
        out = {}

        for c in df["td_or_asd"].unique():
            subset = df[df["td_or_asd"] == c]["free_response_TDprof_norm"].dropna()

            all_text = " ".join(subset.astype(str)).lower().split()
            from collections import Counter
            counts = Counter(all_text)

            out[f"class_{c}"] = {
                "total_texts": len(subset),
                "total_words": len(all_text),
                "unique_words": len(set(all_text)),
                "top_words": dict(counts.most_common(20)),
                "avg_text_length": float(subset.str.len().mean() if len(subset) else 0),
            }

        return out

    # ---------------------------------------------------------
    # MAIN REPORT BUILDER
    # ---------------------------------------------------------
    def generate_feature_contribution_report(self, X, y):
        td_avg, asd_avg = self.analyze_td_vs_asd_feature_means(X, y)
        td_char, asd_char = self.analyze_td_vs_asd_characteristics(X, y)

        return {
            "model_performance": {
                "total_features": len(X.columns),
                "sample_size": len(X),
            },
            "characteristic_analysis": self.analyze_characteristic_contributions(X, y),
            "top_features": self.get_top_discriminative_features(),
            "td_vs_asd_feature_means": {   # NEW — used by visualizer
                "td": td_avg,
                "asd": asd_avg,
            },
            "td_vs_asd_patterns": {        # OLD structure kept for compatibility
                "td": td_char,
                "asd": asd_char,
            },
        }

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------
    def save_analysis_results(self, analysis, filename=f"explainability_analysis_{model_version}.json"):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        out_dir = project_root / "Results" / model_version / "explainability"
        out_dir.mkdir(parents=True, exist_ok=True)

        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        path = out_dir / filename
        with open(path, "w") as f:
            json.dump(convert(analysis), f, indent=2)

        print(f"Explainability saved to: {path}")

    # ---------------------------------------------------------
    # RUN EVERYTHING
    # ---------------------------------------------------------
def run_explainability_analysis():
    analyzer = ExplainabilityAnalyzer()

    print("Loading model & data...")
    df, X, y = analyzer.load_model_and_data()

    print("Generating contribution report...")
    analysis = analyzer.generate_feature_contribution_report(X, y)

    print("Analyzing text...")
    analysis["text_patterns"] = analyzer.analyze_text_patterns_by_class(df)

    print("Saving...")
    analyzer.save_analysis_results(analysis)

    print("Explainability complete.")
    return analysis


if __name__ == "__main__":
    run_explainability_analysis()
