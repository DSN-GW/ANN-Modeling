import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

from .xgboost_model import XGBoostClassifier

data_version = "Data_v4"
model_version = "V4"

# ---------------------------------------------------------
# MULTI-FORMAT SAVE UTILITY  (ADDED)
# ---------------------------------------------------------
def save_figure_multi_formats(fig, base_path, formats=("png", "pdf", "svg")):
    base_path = Path(base_path)
    for ext in formats:
        fig.savefig(base_path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


def _results_dir() -> Path:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    rd = project_root / "Results" / data_version / model_version / "explainability" / "lime"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def _load_model_and_data():
    clf = XGBoostClassifier()
    clf.load_model()
    df = clf.load_data()
    X, y = clf.prepare_features(df)

    # LIME wants raw features; we wrap predict_proba to apply scaling inside
    def predict_proba_raw(Xraw: np.ndarray) -> np.ndarray:
        Xdf = pd.DataFrame(Xraw, columns=clf.feature_names)
        Xscaled = clf.scaler.transform(Xdf)
        return clf.model.predict_proba(Xscaled)

    return clf, X, y, predict_proba_raw


def _positive_class_index(model) -> int:
    try:
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return classes.index(1)
        return len(classes) - 1
    except Exception:
        return 1


def explain_global(
    num_samples: int = 2000,
    num_features: int = 15,
    num_explanations: int = 100,
    random_state: int = 42,
):
    rd = _results_dir()
    clf, X, y, predict_fn = _load_model_and_data()

    if len(X) > num_samples:
        Xs = X.sample(num_samples, random_state=random_state)
    else:
        Xs = X.copy()

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=clf.feature_names,
        class_names=["TD", "ASD"],
        discretize_continuous=True,
        mode="classification",
        random_state=random_state,
    )

    agg = {f: [] for f in clf.feature_names}
    pos_idx = _positive_class_index(clf.model)

    chosen = Xs.sample(min(num_explanations, len(Xs)), random_state=random_state)
    for _, row in chosen.iterrows():
        exp = explainer.explain_instance(row.values, predict_fn, num_features=num_features)
        fmap = exp.as_map()[pos_idx]
        for feat_idx, weight in fmap:
            fname = clf.feature_names[feat_idx]
            agg[fname].append(abs(weight))

    mean_abs = {f: float(np.mean(v)) if len(v) else 0.0 for f, v in agg.items()}
    out_json = rd / f"lime_global_mean_abs_{model_version}.json"
    with open(out_json, "w") as f:
        json.dump(dict(sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)), f, indent=2)

    # ---- GLOBAL BAR CHART (UPDATED TO MULTI-FORMAT SAVE) ----
    top = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)[:20]
    labels, vals = zip(*top) if top else ([], [])

    fig = plt.figure(figsize=(12, 7))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |LIME weight|")
    plt.title(f"LIME Global (aggregated) - {model_version}")
    plt.tight_layout()

    save_figure_multi_formats(fig, rd / f"lime_global_bar_{model_version}")
    plt.close(fig)

    return {"outputs": [str(out_json)]}


def explain_local(indices: Optional[Sequence[int]] = None, num_features: int = 15, random_state: int = 0):
    rd = _results_dir()
    clf, X, y, predict_fn = _load_model_and_data()

    if indices is None:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=min(3, len(X)), replace=False).tolist()

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=clf.feature_names,
        class_names=["TD", "ASD"],
        discretize_continuous=True,
        mode="classification",
        random_state=random_state,
    )

    artifacts = []
    for idx in indices:
        row = X.iloc[int(idx)]
        exp = explainer.explain_instance(row.values, predict_fn, num_features=num_features)

        # HTML stays unchanged
        html_path = rd / f"lime_local_idx{idx}_{model_version}.html"
        exp.save_to_file(str(html_path))

        # ---- STATIC PNG/PDF/SVG (UPDATED) ----
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(8, 5)
        plt.tight_layout()

        save_figure_multi_formats(fig, rd / f"lime_local_idx{idx}_{model_version}")
        plt.close(fig)

        artifacts.append({
            "index": int(idx),
            "html": str(html_path),
            "png": str(html_path.with_suffix(".png"))
        })

    return artifacts


if __name__ == "__main__":
    print("Running LIME global/local...")
    print(explain_global())
    print(explain_local())
