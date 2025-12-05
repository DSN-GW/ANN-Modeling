import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

from .xgboost_model import XGBoostClassifier
from .predict import ModelPredictor

data_version = "Data_v7-3"
model_version = "V7-3"


def _results_dir() -> Path:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    rd = project_root / "Results"  / model_version / "explainability" / "lime"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def _load_model_and_data():
    clf = XGBoostClassifier()
    clf.load_model()
    df = clf.load_data()
    X, y = clf.prepare_features(df)
    predictor = ModelPredictor()

    def predict_proba_raw(Xraw: np.ndarray) -> np.ndarray:
        Xdf = pd.DataFrame(Xraw, columns=clf.feature_names)
        Xscaled = clf.scaler.transform(Xdf)
        return clf.model.predict_proba(Xscaled)

    return clf, X, y, predict_proba_raw, predictor


def _positive_class_index(model) -> int:
    """Best effort to pick the ASD/positive class index for LIME maps."""
    try:
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return classes.index(1)
        # fallback: last class as positive
        return len(classes) - 1
    except Exception:
        return 1  # generic binary fallback


def explain_global(
    num_samples: int = 2000,
    num_features: int = 15,
    num_explanations: int = 100,
    random_state: int = 42,
):
    """
    Aggregate LIME weights across many instances into a crude 'global' view.
    Uses exp.as_map() (feature indices) to avoid bin-string keys.
    Saves:
      - aggregated mean |weight| per feature (JSON)
      - top-N bar chart (PNG)
    """
    rd = _results_dir()
    clf, X, y, predict_fn, _ = _load_model_and_data()

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

    # aggregate by feature *index* to avoid bin label strings
    agg = {f: [] for f in clf.feature_names}
    pos_idx = _positive_class_index(clf.model)

    chosen = Xs.sample(min(num_explanations, len(Xs)), random_state=random_state)
    for _, row in chosen.iterrows():
        exp = explainer.explain_instance(row.values, predict_fn, num_features=num_features)
        # exp.as_map()[class_index] -> List[(feature_index, weight)]
        fmap = exp.as_map()[pos_idx]
        for feat_idx, weight in fmap:
            fname = clf.feature_names[feat_idx]
            agg[fname].append(abs(weight))

    mean_abs = {f: float(np.mean(v)) if len(v) else 0.0 for f, v in agg.items()}
    out_json = rd / f"lime_global_mean_abs_{model_version}.json"
    with open(out_json, "w") as f:
        json.dump(dict(sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)), f, indent=2)

    # Plot top 20
    top = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)[:20]
    labels, vals = zip(*top) if top else ([], [])
    plt.figure(figsize=(12, 7))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |LIME weight|")
    plt.title(f"LIME Global (aggregated) - {model_version}")
    plt.tight_layout()
    out_png = rd / f"lime_global_bar_{model_version}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return {"outputs": [str(out_json), str(out_png)]}


def explain_local(indices: Optional[Sequence[int]] = None, num_features: int = 15, random_state: int = 0, dataset: str = "train", df = False):
    """
    Generate per-instance LIME HTML explanations (and quick PNG dumps).
    Returns list of artifacts per index.
    """
    rd = _results_dir()
    clf, X, y, predict_fn, predictor = _load_model_and_data()
    X_ = X

    if dataset == "test":
        if not df:
            print("here")
            X_ = predictor.preprocess_new_data(df,True)
            X_, y = clf.prepare_features(X_)
        else:
            print("there")
            X_ = predictor.preprocess_new_data(df,False)
            X_, y = clf.prepare_features(X_)


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
        row = X_.iloc[int(idx)]
        row_df = pd.DataFrame([row], columns=X.columns)
        X_scaled = clf.scaler.transform(row_df)
        proba = clf.model.predict_proba(X_scaled)[0]
        pred_class = int(np.argmax(proba))

        exp = explainer.explain_instance(row.values, predict_fn, num_features=num_features)
        html_path = rd / f"lime_local_idx{idx}_{model_version}.html"
        exp.save_to_file(str(html_path))

        # Also save a quick static figure
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(8, 5)
        plt.tight_layout()
        png_path = rd / f"lime_local_idx{idx}_{model_version}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        artifacts.append({"index": int(idx), "html": str(html_path), "png": str(png_path), "predicted_class": pred_class,
            "prediction_proba": proba.tolist(), "weights": exp.as_list()})
    return artifacts


if __name__ == "__main__":
    print("Running LIME global/local...")
    print(explain_global())
    print(explain_local())
