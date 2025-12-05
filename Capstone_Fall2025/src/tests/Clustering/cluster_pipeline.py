import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score

from consensusclustering import ConsensusClustering

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.stats import f_oneway, kruskal
import scikit_posthocs as sp

from umap.umap_ import UMAP
from kneed import KneeLocator

import nltk
from textblob import TextBlob
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# NLTK setup
# -------------------------------------------------------------------
def _nltk_setup():
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

_nltk_setup()


# ================================================================
# 0. NLP feature extractor
# ================================================================
class NLPFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        self.positive_words = {
            "like","likes","love","loves","enjoy","enjoys","good","great",
            "awesome","amazing","wonderful","fantastic","excellent","perfect",
            "happy","fun","cool","nice","best","favorite","prefer","prefers"
        }
        self.negative_words = {
            "dislike","dislikes","hate","hates","bad","terrible","awful",
            "horrible","worst","not","never","no","dont","don't","doesnt",
            "doesn't","cannot","can't","wont","won't"
        }

    def get_empty_nlp_features(self):
        return {
            "word_count": 0,
            "sentence_count": 0,
            "char_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "shortness_score": 1,
            "lexical_diversity": 0,
            "sentiment_polarity": 0,
            "sentiment_subjectivity": 0,
            "positive_word_count": 0,
            "negative_word_count": 0,
            "positive_word_ratio": 0,
            "negative_word_ratio": 0,
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
        }

    def extract_nlp_features(self, text):
        if pd.isna(text) or str(text).strip() == "":
            return self.get_empty_nlp_features()

        try:
            text = str(text).lower()
            words = word_tokenize(text)
            sentences = sent_tokenize(text)

            feats = {}
            feats["word_count"] = len(words)
            feats["sentence_count"] = len(sentences)
            feats["char_count"] = len(text)
            feats["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
            feats["avg_sentence_length"] = (len(words) / len(sentences)) if sentences else 0

            # FSR-style shortness proxy
            feats["shortness_score"] = 1 / (1 + feats["word_count"])

            unique_words = set(words)
            feats["lexical_diversity"] = (len(unique_words) / len(words)) if words else 0

            blob = TextBlob(text)
            feats["sentiment_polarity"] = blob.sentiment.polarity
            feats["sentiment_subjectivity"] = blob.sentiment.subjectivity

            pos_ct = sum(w in self.positive_words for w in words)
            neg_ct = sum(w in self.negative_words for w in words)
            feats["positive_word_count"] = pos_ct
            feats["negative_word_count"] = neg_ct
            feats["positive_word_ratio"] = pos_ct / len(words) if words else 0
            feats["negative_word_ratio"] = neg_ct / len(words) if words else 0

            try:
                feats["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
                feats["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            except Exception:
                feats["flesch_reading_ease"] = 0
                feats["flesch_kincaid_grade"] = 0

            return feats
        except Exception:
            return self.get_empty_nlp_features()


# ================================================================
# 1. EMBEDDING
# ================================================================
def embed_data(
    X_scaled,
    method="pca",     # "pca" or "pca_umap"
    pca_var=0.95,
    umap_n_neighbors=20,
    umap_min_dist=0.1,
    umap_n_components=10,
    random_state=42
):
    """
    Fits PCA based on threshold supplied (default is 0.95) then UMAP (optional).

    Returns:
        X_embed, pca_model, umap_model (or None)
    """
    pca_full = PCA().fit(X_scaled)
    k = np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), pca_var) + 1

    pca_model = PCA(n_components=k, random_state=random_state)
    X_pca = pca_model.fit_transform(X_scaled)

    if method == "pca":
        return X_pca, pca_model, None

    n_comp = min(umap_n_components, X_pca.shape[1])
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=n_comp,
        metric="euclidean",
        random_state=random_state
    )
    X_umap = umap_model.fit_transform(X_pca)
    return X_umap, pca_model, umap_model


def fit_umap_2d(X_scaled, random_state=42):
    umap2 = UMAP(
        n_neighbors=20, min_dist=0.1, n_components=2,
        metric="euclidean", random_state=random_state
    )
    return umap2.fit_transform(X_scaled)


# ================================================================
# 4. Consensus Matrix, PAC, ΔAUC
# ================================================================
def compute_pac(consensus_matrix, lower=0.1, upper=0.9):
    """
    PAC score = CDF(upper) - CDF(lower). Lower PAC = better clustering.
    """
    flat = consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)]
    hist, _ = np.histogram(flat, bins=100, range=(0, 1), density=True)
    cdf = np.cumsum(hist) / np.sum(hist)

    idx_lower = max(int(lower * 100) - 1, 0)
    idx_upper = max(int(upper * 100) - 1, 0)

    return cdf[idx_upper] - cdf[idx_lower]


def compute_delta_auc(cdf_area_dict):
    """
    Monti-style proportional ΔAUC between successive Ks.
    """
    ks = sorted(cdf_area_dict.keys())
    aucs = np.array([cdf_area_dict[k] for k in ks])

    delta = np.zeros_like(aucs)
    for i in range(1, len(ks)):
        # proportional improvement
        if aucs[i] != 0:
            delta[i] = (aucs[i] - aucs[i - 1]) / aucs[i]
        else:
            delta[i] = 0.0

    delta_dict = {k: d for k, d in zip(ks, delta)}
    return delta_dict


def consensus_clustering(X, minK=2, maxK=12, reps=500, pItem=0.8):
    """
    Run consensus clustering using the `consensusclustering` Python package
    on an embedded matrix X (e.g., PCA or PCA+UMAP space).
    """
    X = np.asarray(X)

    base = KMeans(n_init="auto", random_state=RANDOM_STATE)

    cc = ConsensusClustering(
        clustering_obj=base,
        min_clusters=minK,
        max_clusters=maxK,
        n_resamples=reps,
        resample_frac=pItem,
        k_param="n_clusters",
        rng=RANDOM_STATE
    )

    cc.fit(X, progress_bar=True, n_jobs=-1)

    Ks = cc.cluster_range_
    consensus_mats = {}
    pac_scores = {}
    avg_consensus_scores = {}
    cdf_area_scores = {}

    # Store consensus data
    for k in Ks:
        M = cc.consensus_k(k)
        consensus_mats[k] = M
        pac_scores[k] = compute_pac(M)
        avg_consensus_scores[k] = np.mean(M)
        cdf_area_scores[k] = cc.area_under_cdf(k)
        cc.plot_clustermap(k)

    # ---- Plots ----
    print("\nPlotting CDF curves...")
    cc.plot_cdf()
    plt.title("Consensus CDF Curves")
    plt.show()

    print("Plotting AUC (change in area) curve...")
    cc.plot_change_area_under_cdf()
    plt.title("Delta AUC Curve")
    plt.show()

    print("Plotting AUC(CDF) vs K with knee...")
    cc.plot_auc_cdf(include_knee=True)
    plt.title("AUC(CDF) vs K")
    plt.show()

    return {
        "cc": cc,
        "consensus_mats": consensus_mats,
        "pac": pac_scores,
        "avg_consensus": avg_consensus_scores,
        "cdf_area": cdf_area_scores,
        "K_range": Ks
    }


# ================================================================
# 5. BEST K Selection Methods
# ================================================================
def best_k_pac(pac_dict):
    """Choose k with minimal PAC."""
    return min(pac_dict, key=pac_dict.get)


def best_k_stability(stability_dict):
    """Choose k with maximum mean-consensus."""
    return max(stability_dict, key=stability_dict.get)


def kneedle_best_k_pac(pac_dict):
    """
    PAC(k): usually decreasing and roughly concave.
    Use concave + decreasing.
    """
    ks = np.array(sorted(pac_dict.keys()))
    vals = np.array([pac_dict[k] for k in ks])

    kneedle = KneeLocator(
        ks, vals,
        curve="concave",
        direction="decreasing"
    )
    return kneedle.knee


def kneedle_best_k_cdf_area(cdf_area_dict):
    """
    AUC(CDF)(k): increasing and concave (saturating).
    Use concave + increasing.
    """
    ks = np.array(sorted(cdf_area_dict.keys()))
    vals = np.array([cdf_area_dict[k] for k in ks])

    kneedle = KneeLocator(
        ks, vals,
        curve="concave",
        direction="increasing"
    )
    return kneedle.knee


def kneedle_best_k_delta_auc(delta_auc_dict):
    """
    ΔAUC(k): typically decreasing, convex (big improvements at small k, then flatten).
    Use convex + decreasing.
    """
    ks = np.array(sorted(delta_auc_dict.keys()))
    vals = np.array([delta_auc_dict[k] for k in ks])

    kneedle = KneeLocator(
        ks, vals,
        curve="convex",
        direction="decreasing"
    )
    return kneedle.knee


def kneedle_best_k_stability(stability_dict):
    """
    Stability(k) = mean consensus: increasing concave (saturating).
    Use concave + increasing.
    """
    ks = np.array(sorted(stability_dict.keys()))
    vals = np.array([stability_dict[k] for k in ks])

    kneedle = KneeLocator(
        ks, vals,
        curve="concave",
        direction="increasing"
    )
    return kneedle.knee


def choose_best_k(results):
    """
    Aggregate all best-K criteria:
      - PAC minimum
      - PAC Kneedle elbow
      - Stability maximum
      - Stability Kneedle elbow
      - AUC(CDF) Kneedle elbow
      - ΔAUC Kneedle elbow
    """
    pac = results["pac"]
    stab = results["avg_consensus"]
    auc_area = results["cdf_area"]

    delta_auc = compute_delta_auc(auc_area)

    k1 = best_k_pac(pac)
    k2 = kneedle_best_k_pac(pac)
    k3 = best_k_stability(stab)
    k4 = kneedle_best_k_stability(stab)
    k5 = kneedle_best_k_cdf_area(auc_area)
    k6 = kneedle_best_k_delta_auc(delta_auc)

    print("\n============== BEST-K SUMMARY ===============")
    print(f"1. PAC minimum                 k = {k1}")
    print(f"2. PAC Kneedle elbow           k = {k2}")
    print(f"3. Stability maximum           k = {k3}")
    print(f"4. Stability Kneedle elbow     k = {k4}")
    print(f"5. AUC(CDF) Kneedle elbow      k = {k5}")
    print(f"6. ΔAUC Kneedle elbow          k = {k6}")

    # Choose the smallest reasonable K among elbow-based methods
    candidates = [k for k in [k2, k4, k5, k6] if k is not None]

    if len(candidates):
        final = min(candidates)
    else:
        final = k1  # fallback to PAC minimum

    print(f"\n>>> FINAL recommended k: {final}")
    print("=============================================\n")

    return {
        "pac_min": k1,
        "pac_kneedle": k2,
        "stab_max": k3,
        "stab_kneedle": k4,
        "auc_kneedle": k5,
        "delta_auc_kneedle": k6,
        "final": final
    }


# ================================================================
# 6. CONSENSUS CLUSTERING + FINAL LABELS
# ================================================================
def consensus_clusterer(consensus, k):
    """
    Final consensus clustering on distance D = 1 - consensus.
    """
    D = 1 - consensus
    labels = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average"
    ).fit_predict(D)
    return labels


# ================================================================
# 7. PLOTTING HELPERS
# ================================================================
def plot_umap_clusters(points_2d, labels, title):
    plt.figure(figsize=(9, 7))
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab20")

    for i, lab in enumerate(uniq):
        m = labels == lab
        plt.scatter(points_2d[m, 0], points_2d[m, 1],
                    s=18, alpha=0.85, label=f"cluster {lab}",
                    c=[cmap(i % 10)])

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_pca_loadings(pca_model, feature_names, top_n=10):
    """
    Plots loadings for each PC (top_n strongest features)
    and prints the numeric loading value on the bars.
    """
    loadings = pca_model.components_
    n_pcs = loadings.shape[0]

    for pc in range(min(n_pcs, 5)):  # show first 5 PCs
        comp = loadings[pc]

        abs_idx = np.argsort(np.abs(comp))[::-1][:top_n]
        sel_features = np.array(feature_names)[abs_idx]
        sel_values = comp[abs_idx]

        plt.figure(figsize=(12, 4))
        bars = plt.bar(sel_features, sel_values)

        for bar, value in zip(bars, sel_values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"PCA Loadings: PC{pc+1} (Top {top_n} features)")
        plt.tight_layout()
        plt.show()


def plot_pac_curve(pac_dict):
    ks = sorted(pac_dict.keys())
    vals = [pac_dict[k] for k in ks]
    plt.figure(figsize=(7, 4))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("PAC")
    plt.title("PAC(k) — lower is better")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stability_curve(stability_dict):
    ks = sorted(stability_dict.keys())
    vals = [stability_dict[k] for k in ks]
    plt.figure(figsize=(7, 4))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("Cluster Stability (mean consensus)")
    plt.title("Mean Stability vs k — higher is better")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ================================================================
# 8. CLUSTER PROFILES + SIGNIFICANCE
# ================================================================
def cluster_profiles(df_asd, features, label_col="consensus_cluster",
                     outdir="cluster_profiles"):
    """
    Creates per feature stats : mean, median and std - for each consensus cluster.
    """
    os.makedirs(outdir, exist_ok=True)

    results = {}

    for feat in features:
        df_feat = (
            df_asd.groupby(label_col)[feat]
                  .agg(["count", "mean", "std", "median"])
                  .sort_index()
        )
        results[feat] = df_feat

        fig, ax = plt.subplots(figsize=(10, 0.8 + 0.4 * len(df_feat)))
        ax.axis("off")

        plt.text(
            0.5, 1.01,
            f"Cluster Profile for Feature: {feat}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
            transform=ax.transAxes
        )

        table = ax.table(
            cellText=df_feat.round(3).values,
            colLabels=df_feat.columns,
            rowLabels=df_feat.index,
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        fname = os.path.join(outdir, f"profile_{feat}.pdf")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        print(f"Saved : {fname}")

    return results


def significance_tests(df_asd, features, label_col="consensus_cluster",
                       _dir="significance_tests"):
    """
    Significance tests:
       • ANOVA + Kruskal
       • Dunn post-hoc
    """
    os.makedirs(_dir, exist_ok=True)

    print("\n=== Significance tests across consensus clusters ===")
    clusters = sorted(df_asd[label_col].unique())

    summary_rows = []

    # ANOVA + Kruskal
    for f in features:
        groups = [df_asd[df_asd[label_col] == c][f].values for c in clusters]

        try:
            F, p_a = f_oneway(*groups)
        except Exception:
            F, p_a = np.nan, np.nan

        try:
            H, p_k = kruskal(*groups)
        except Exception:
            H, p_k = np.nan, np.nan

        print(f"\n---Feature: {f}---")
        print(f"  ANOVA:   F={F:.3f}, p={p_a:.3g}")
        print(f"  Kruskal: H={H:.3f}, p={p_k:.3g}")

        summary_rows.append({
            "feature": f,
            "anova_F": F,
            "anova_p": p_a,
            "kruskal_H": H,
            "kruskal_p": p_k
        })

    summary_df = pd.DataFrame(summary_rows).round(3)

    fig, ax = plt.subplots(figsize=(12, 0.7 + 0.3 * len(summary_df)))
    ax.axis("off")

    plt.text(
        0.5, 1.01,
        "One Way ANOVA and Kruskal Test Results for Consensus Clusters",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold",
        transform=ax.transAxes
    )

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    fname = os.path.join(_dir, "significance.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"\nSaved significance test summary : {fname}")

    # Dunn post-hoc test
    print("\n=== Dunn posthoc Test for Pairwise Clusters ===")

    for f in features:
        dunn = sp.posthoc_dunn(
            df_asd, val_col=f, group_col=label_col, p_adjust="bonferroni"
        )

        print(f"\nDunn p-values for {f}:")
        print(dunn)

        dunn_rounded = dunn.round(3)

        fig, ax = plt.subplots(figsize=(12, 0.7 + 0.3 * len(dunn_rounded)))
        ax.axis("off")

        plt.text(
            0.5, 1.01,
            f"Dunn Posthoc Test for Feature: {f}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
            transform=ax.transAxes
        )

        table = ax.table(
            cellText=dunn_rounded.values,
            colLabels=dunn_rounded.columns.astype(str),
            rowLabels=dunn_rounded.index.astype(str),
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        fname = os.path.join(_dir, f"dunn_{f}.pdf")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        print(f" Saved Dunn matrix : {fname}")


# ================================================================
# 9. TD PROJECTION + CONTAMINATION
# ================================================================
def train_asd_consensus_classifier(X_embed, consensus_labels):
    clf = RandomForestClassifier(
        n_estimators=400,
        bootstrap=True,
        random_state=42
    )
    clf.fit(X_embed, consensus_labels)
    return clf


def project_td_to_consensus(
    df,
    features,
    scaler,
    pca_model,
    umap_model,
    clf
):
    df_td = df[df["td_or_asd"] == 0].reset_index(drop=True)
    if df_td.empty:
        return None, None

    X_td_scaled = scaler.transform(df_td[features].values)
    X_td_pca = pca_model.transform(X_td_scaled)
    X_td_embed = umap_model.transform(X_td_pca) if umap_model is not None else X_td_pca

    df_td = df_td.copy()
    df_td["consensus_cluster_pred"] = clf.predict(X_td_embed)
    return df_td, X_td_embed


def td_contamination(df_asd, df_td, label_col="consensus_cluster", pdf_filename="TD_Contamination.pdf"):
    asd_counts = df_asd[label_col].value_counts().sort_index()
    td_counts = df_td["consensus_cluster_pred"].value_counts().sort_index()

    all_clusters = sorted(set(asd_counts.index).union(td_counts.index))
    rows = []
    for c in all_clusters:
        asd_n = int(asd_counts.get(c, 0))
        td_n = int(td_counts.get(c, 0))
        total = asd_n + td_n
        frac = td_n / total if total > 0 else np.nan
        rows.append({
            "cluster": c,
            "ASD_count": asd_n,
            "TD_count": td_n,
            "total": total,
            "td_fraction": frac
        })

    df_ratio = pd.DataFrame(rows)
    print("\n=== TD Contamination per Consensus Cluster ===")
    print(df_ratio)

    df_round = df_ratio.round(3)

    fig, ax = plt.subplots(figsize=(12, 0.7 + 0.3 * len(df_round)))
    ax.axis("off")

    table = ax.table(
        cellText=df_round.values,
        colLabels=df_round.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.close()

    print(f"\n Saved TD contamination table : {pdf_filename}")
    return df_ratio


# ================================================================
# 10. MAIN
# ================================================================
if __name__ == "__main__":

    # ---------------- USER CONFIG ----------------
    current_dir = Path(__file__).parent
    data_path = current_dir / "LLM data_aggregate.csv"
    TEXT_COLUMN = "free_response_TDprof_norm"

    FEATURES = [
        'FSR',
        'TDNorm_avg_PE',
        'TDnorm_concept_learning',
        'overall_concept_learning',
        # 'word_count',
        # 'avg_word_length',
        # 'avg_sentence_length',
        # 'shortness_score',
        # 'sentiment_polarity',
        # 'sentiment_subjectivity',
        # 'negative_word_count',
        # 'positive_word_ratio',
        # 'negative_word_ratio',
        # 'flesch_reading_ease'
    ]

    COLS = ["sub", "td_or_asd"] + FEATURES
    EMBED_METHOD = "pca"   # "pca" or "pca_umap"

    PCA_VAR = 0.95
    UMAP_N_NEIGHBORS = 20
    UMAP_MIN_DIST = 0.1
    UMAP_N_COMPONENTS = 10

    RANDOM_STATE = 42

    # ---------------------------------------------
    df = pd.read_csv(data_path)
    print(f"Loaded raw data: {df.shape} from {data_path}")

    # Extract NLP features
    nlp_extractor = NLPFeatureExtractor()
    print(f"Extracting NLP features from `{TEXT_COLUMN}`...")
    nlp_features = [nlp_extractor.extract_nlp_features(t) for t in df[TEXT_COLUMN]]
    nlp_df = pd.DataFrame(nlp_features)
    df = pd.concat([df.reset_index(drop=True), nlp_df.reset_index(drop=True)], axis=1)
    df = df[COLS]
    df.dropna(inplace=True)

    # ASD-only for clustering
    df_asd = df[df["td_or_asd"] == 1].reset_index(drop=True)
    print("ASD-only shape:", df_asd.shape)
    print("TD-only shape:", df[df["td_or_asd"] == 0].shape)

    # Scale ASD
    scaler = StandardScaler()
    X_asd = df_asd[FEATURES].values
    X_asd_scaled = scaler.fit_transform(X_asd)

    # Embed ASD (PCA / PCA + UMAP)
    X_asd_embed, pca_model, umap_model = embed_data(
        X_asd_scaled,
        method=EMBED_METHOD,
        pca_var=PCA_VAR,
        umap_n_neighbors=UMAP_N_NEIGHBORS,
        umap_min_dist=UMAP_MIN_DIST,
        umap_n_components=UMAP_N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    # PCA loadings
    plot_pca_loadings(pca_model, FEATURES, top_n=10)

    # ============================================================
    # FULL CONSENSUS CLUSTERING OVER K (Monti + Şenbabaoğlu)
    # ============================================================
    cc_dict = consensus_clustering(
        X_asd_embed,
        minK=2,
        maxK=12,
        reps=500,
        pItem=0.8
    )

    # Additional metric plots
    plot_pac_curve(cc_dict["pac"])
    plot_stability_curve(cc_dict["avg_consensus"])

    # Choose best K 
    best_k_info = choose_best_k(cc_dict)
    best_k = best_k_info["final"]
    best_k = 5

    # Extract best consensus matrix and final consensus labels
    consensus_best = cc_dict["consensus_mats"][best_k]
    consensus_labels = consensus_clusterer(consensus_best, best_k)

    # Attach labels to ASD dataframe
    df_asd = df_asd.copy()
    df_asd["consensus_cluster"] = consensus_labels

    # Silhouette on consensus distance
    sil = silhouette_score(1 - consensus_best, consensus_labels, metric="precomputed")
    print(f"\nSilhouette (consensus distance, k={best_k}): {sil:.3f}")

    # ============================================================
    # CLUSTER PROFILES + SIGNIFICANCE
    # ============================================================
    prof = cluster_profiles(df_asd, FEATURES)
    print("\n=== Consensus cluster profiles ===")
    for feat, table in prof.items():
        print("\n", feat)
        print(table)

    significance_tests(df_asd, FEATURES)

    # ============================================================
    # TD PROJECTION + CONTAMINATION
    # ============================================================
    clf = train_asd_consensus_classifier(X_asd_embed, consensus_labels)
    df_td, _ = project_td_to_consensus(
        df, FEATURES, scaler, pca_model, umap_model, clf
    )
    if df_td is not None:
        contam_df = td_contamination(df_asd, df_td, label_col="consensus_cluster")

        plt.figure(figsize=(6, 4))
        sns.barplot(data=contam_df, x="cluster", y="td_fraction")
        plt.title("TD contamination fraction per consensus cluster")
        plt.tight_layout()
        plt.show()

        df_td.to_csv("td_projected_to_consensus.csv", index=False)
        print("\nSaved TD projections: td_projected_to_consensus.csv")

        contamination_rounded = contam_df.round(5)

        fig, ax = plt.subplots(figsize=(12, 0.7 + 0.3 * len(contamination_rounded)))
        ax.axis("off")

        plt.text(
            0.5, 1.01,
            "TD Contamination Results:",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
            transform=ax.transAxes
        )

        table = ax.table(
            cellText=contamination_rounded.values,
            colLabels=contamination_rounded.columns.astype(str),
            rowLabels=contamination_rounded.index.astype(str),
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        fname = "TD_Contamination_Report.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

    # Save ASD-labeled output
    df_asd.to_csv("asd_consensus_labeled.csv", index=False)
    print("\nSaved ASD consensus labels: asd_consensus_labeled.csv")
