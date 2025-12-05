import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .xgboost_model import XGBoostClassifier
from .predict import ModelPredictor

data_version = "Data_v7-2"
model_version = "V7-2"

def _results_dir() -> Path:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    rd = project_root / "Results" / data_version / model_version / "explainability" / "shap"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def _load_model_and_data() :
    clf = XGBoostClassifier()
    clf.load_model()

    df = clf.load_data()
    X, y = clf.prepare_features(df)
    predictor = ModelPredictor()

    X_scaled = pd.DataFrame(clf.scaler.transform(X), columns=clf.feature_names, index=X.index)
    return clf, X, X_scaled, y, predictor


def explain_global(sample_size: int = 2000, random_state: int = 42) -> dict:
    """
    Creates global SHAP explanations:
      - summary beeswarm (PNG)
      - summary bar plot (PNG)
      - saves mean |SHAP| per feature to JSON
    """
    rd = _results_dir()
    clf, X, Xs, _, _ = _load_model_and_data()

    if sample_size and len(Xs) > sample_size:
        Xs = Xs.sample(sample_size, random_state=random_state)

    explainer = shap.TreeExplainer(clf.model)
    shap_values = explainer.shap_values(Xs)

    # Save mean |SHAP| stats
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_abs_dict = {feat: float(val) for feat, val in zip(clf.feature_names, mean_abs)}
    with open(rd / f"shap_global_mean_abs_{model_version}.json", "w") as f:
        json.dump(dict(sorted(mean_abs_dict.items(), key=lambda kv: kv[1], reverse=True)), f, indent=2)

    # Beeswarm
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    plt.savefig(rd / f"shap_summary_beeswarm_{model_version}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bar
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, Xs, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(rd / f"shap_summary_bar_{model_version}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "samples_used": int(len(Xs)),
        "outputs": [
            str(rd / f"shap_summary_beeswarm_{model_version}.png"),
            str(rd / f"shap_summary_bar_{model_version}.png"),
            str(rd / f"shap_global_mean_abs_{model_version}.json"),
        ],
    }


def explain_local(indices: Optional[Sequence[int]] = None, top_n: int = 15, random_state: int = 0,dataset: str = "train", df = False) -> list[dict]:
    """
    Creates SHAP waterfall plots for specific row indices
    If indices is None, pick 3 random rows.
    """
    rd = _results_dir()
    print("We have done this")
    clf, X, Xs, _ , predictor= _load_model_and_data()

    if dataset == "test":
        print("Inside Test")
        if not df:
            print("here-shap")
            X = predictor.preprocess_new_data(df, True)
            print(X.head(5))
            X, y = clf.prepare_features(X)
            Xs = pd.DataFrame(clf.scaler.transform(X), columns=clf.feature_names, index=X.index)
            print("Xs Generated")
        else:
            print("there-shap")
            X = predictor.preprocess_new_data(df, False)
            X, y = clf.prepare_features(X)
            Xs = pd.DataFrame(clf.scaler.transform(X), columns=clf.feature_names, index=X.index)

    if indices is None:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(Xs), size=min(3, len(Xs)), replace=False).tolist()

    explainer = shap.TreeExplainer(clf.model)
    sv = explainer.shap_values(Xs)

    out = []
    for idx in indices:
        print("In the loop")
        row = Xs.iloc[int(idx)]
        shap_row = sv[int(idx)]
        base_value = float(explainer.expected_value)
        row_df = pd.DataFrame([row], columns=Xs.columns)
        proba = clf.model.predict_proba(row_df)[0]
        pred_class = int(np.argmax(proba))

        # Waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            base_value, shap_row, feature_names=list(Xs.columns), max_display=top_n, show=False
        )
        plt.tight_layout()
        png_path = rd / f"shap_local_idx{idx}_{model_version}.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()

        out.append({"index": int(idx), "plot": str(png_path),
                    "predicted_class": pred_class,
                    "prediction_proba": proba,
                    "base_value": base_value,
                    "shap_values": shap_row.tolist(),
                    "feature_values": X.iloc[int(idx)].to_dict(),
                    })

    return out


if __name__ == "__main__":
    g = explain_global()
    print("Global SHAP:", g)
    l = explain_local()
    print("Local SHAP:", l)
