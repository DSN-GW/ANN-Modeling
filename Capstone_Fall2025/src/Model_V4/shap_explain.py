import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .xgboost_model import XGBoostClassifier
from visualization import save_figure_multi_formats

data_version = "Data_v4"
model_version = "V4"


def _results_dir() -> Path:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    rd = project_root / "Results" / data_version / model_version / "explainability" / "shap"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def _load_model_and_data() -> tuple[XGBoostClassifier, pd.DataFrame, pd.DataFrame, pd.Series]:
    clf = XGBoostClassifier()
    clf.load_model()

    df = clf.load_data()
    X, y = clf.prepare_features(df)

    X_scaled = pd.DataFrame(clf.scaler.transform(X), columns=clf.feature_names, index=X.index)
    return clf, X, X_scaled, y


def explain_global(sample_size: int = 2000, random_state: int = 42) -> dict:
    """
    Creates global SHAP explanations:
      - summary beeswarm
      - summary bar
      - saves mean |SHAP| per feature to JSON
    """
    rd = _results_dir()
    clf, X, Xs, _ = _load_model_and_data()

    if sample_size and len(Xs) > sample_size:
        Xs = Xs.sample(sample_size, random_state=random_state)

    explainer = shap.TreeExplainer(clf.model)
    shap_values = explainer.shap_values(Xs)

    # Save mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_abs_dict = {feat: float(val) for feat, val in zip(clf.feature_names, mean_abs)}

    with open(rd / f"shap_global_mean_abs_{model_version}.json", "w") as f:
        json.dump(
            dict(sorted(mean_abs_dict.items(), key=lambda kv: kv[1], reverse=True)),
            f,
            indent=2,
        )

    # Beeswarm plot
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    save_figure_multi_formats(plt.gcf(), rd / f"shap_summary_beeswarm_{model_version}")
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, Xs, plot_type="bar", show=False)
    plt.tight_layout()
    save_figure_multi_formats(plt.gcf(), rd / f"shap_summary_bar_{model_version}")
    plt.close()

    return {
        "samples_used": int(len(Xs)),
        "outputs": [
            str(rd / f"shap_summary_beeswarm_{model_version}.png"),
            str(rd / f"shap_summary_bar_{model_version}.png"),
            str(rd / f"shap_global_mean_abs_{model_version}.json"),
        ],
    }


def explain_local(indices: Optional[Sequence[int]] = None, top_n: int = 15, random_state: int = 0) -> list[dict]:
    """
    Creates SHAP waterfall plots for specific row indices.
    """
    rd = _results_dir()
    clf, X, Xs, _ = _load_model_and_data()

    if indices is None:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(Xs), size=min(3, len(Xs)), replace=False).tolist()

    explainer = shap.TreeExplainer(clf.model)
    shap_vals = explainer.shap_values(Xs)

    out = []
    for idx in indices:
        row = Xs.iloc[int(idx)]
        shap_row = shap_vals[int(idx)]
        base_value = float(explainer.expected_value)

        plt.figure(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            base_value,
            shap_row,
            feature_names=list(Xs.columns),
            max_display=top_n,
            show=False
        )
        plt.tight_layout()

        base = rd / f"shap_local_idx{idx}_{model_version}"
        save_figure_multi_formats(plt.gcf(), base)
        plt.close()

        out.append({"index": int(idx), "plot": str(base.with_suffix('.png')), "base_value": base_value})

    return out


if __name__ == "__main__":
    g = explain_global()
    print("Global SHAP:", g)
    l = explain_local()
    print("Local SHAP:", l)
