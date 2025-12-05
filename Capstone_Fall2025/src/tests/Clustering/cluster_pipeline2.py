
"""
FINAL ASD CONSENSUS CLUSTERING PIPELINE
--------------------------------------

What this script does:

1) Load full dataset
2) Filter ASD-only for unsupervised clustering
3) Scale ASD features
4) Embed ASD with PCA (optional UMAP)
5) Cluster ASD with HDBSCAN (single run)
6) Per-cluster bootstrap stability
7) Bootstrap co-association matrix
8) Consensus delta curve (k=2...Kmax) and best-k selection
9) Consensus clustering on (1 - coassoc)
10) Dendrogram-ordered consensus heatmap
11) UMAP2D visualization of consensus clusters
12) Cluster profiles and significance tests
13) Train classifier on ASD consensus clusters
14) Project TD into ASD consensus space
15) TD contamination + ASD/TD ratios
16) Save labeled ASD + TD outputs

Requirements:
    numpy, pandas, matplotlib, seaborn, sklearn>=1.4, scipy, umap-learn, hdbscan, tqdm, scikit-posthocs

"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.stats import f_oneway, kruskal
import scikit_posthocs as sp

import hdbscan
from umap.umap_ import UMAP

import warnings
warnings.filterwarnings("ignore")

import nltk
from textblob import TextBlob
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
# 0. NLP feature extractor - From data_preprocessor
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
    method="pca_umap",     # "pca" or "pca_umap"
    pca_var=0.95,
    umap_n_neighbors=20,
    umap_min_dist=0.1,
    umap_n_components=10,
    random_state=42
):
    """
    Fits PCA based on threshold supplied (default is 0.95) then UMAP (Optional).

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
# 2. HDBSCAN CLUSTERING
# ================================================================

def cluster_hdbscan(X_embed, min_cluster_size=10, min_samples=5):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(X_embed)
    return labels, clusterer


# ================================================================
# 3. PER-CLUSTER STABILITY
# ================================================================

def cluster_stability(
    X_scaled,
    labels_orig,
    embed_method="pca_umap",
    B=100,
    pca_var=0.95,
    umap_n_neighbors=20,
    umap_min_dist=0.1,
    umap_n_components=10,
    min_cluster_size=10,
    min_samples=5,
    random_state=42
):
    """
    Stability(cluster c) =
      avg over bootstraps of:
        (# original members of c appearing in bootstrap
         that land in same majority boot cluster)
         : divided by :
        (# original members of c appearing in bootstrap)
    """
    orig_clusters = [c for c in sorted(set(labels_orig)) if c != -1]
    orig_members = {c: np.where(labels_orig == c)[0] for c in orig_clusters}

    n = len(labels_orig)
    rng = np.random.default_rng(random_state)

    stab_scores = {c: [] for c in orig_clusters}

    for b in tqdm(range(B), desc="Cluster Stability bootstraps"):
        boot_idx = rng.choice(n, size=n, replace=True)
        Xb = X_scaled[boot_idx]

        Xb_embed, _, _ = embed_data(
            Xb,
            method=embed_method,
            pca_var=pca_var,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_n_components=umap_n_components,
            random_state=random_state + b + 11
        )

        labels_b, _ = cluster_hdbscan(
            Xb_embed,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )

        # first occurrence of orig index in bootstrap
        first_occ = {}
        for pos, oi in enumerate(boot_idx):
            if oi not in first_occ:
                first_occ[oi] = pos

        for c in orig_clusters:
            members = orig_members[c]
            appearing = [i for i in members if i in first_occ]

            if len(appearing) == 0:
                continue

            boot_labels = [labels_b[first_occ[i]] for i in appearing]
            non_noise = [bl for bl in boot_labels if bl != -1]

            if len(non_noise) == 0:
                stab_scores[c].append(0.0)
                continue

            vals, counts = np.unique(non_noise, return_counts=True)
            majority = vals[np.argmax(counts)]

            stab = np.mean([bl == majority for bl in boot_labels if bl != -1])
            stab_scores[c].append(stab)

    stab_final = {
        c: np.mean(stab_scores[c]) if len(stab_scores[c]) > 0 else np.nan
        for c in orig_clusters
    }
    return stab_final, stab_scores


# ================================================================
# 4. CO-ASSOCIATION MATRIX (BOOTSTRAP CONSENSUS)
# ================================================================

def coassociation_matrix(
    X_scaled,
    embed_method="pca_umap",
    B=100,
    pca_var=0.95,
    umap_n_neighbors=20,
    umap_min_dist=0.1,
    umap_n_components=10,
    min_cluster_size=10,
    min_samples=5,
    random_state=42
):
    n = X_scaled.shape[0]
    rng = np.random.default_rng(random_state)

    coassoc = np.zeros((n, n), float)
    denom   = np.zeros((n, n), float)

    for b in tqdm(range(B), desc="Co-association bootstraps"):
        boot_idx = rng.choice(n, size=n, replace=True)
        Xb = X_scaled[boot_idx]

        Xb_embed, _, _ = embed_data(
            Xb,
            method=embed_method,
            pca_var=pca_var,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_n_components=umap_n_components,
            random_state=random_state + b + 101
        )

        labels_b, _ = cluster_hdbscan(
            Xb_embed,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )

        first_pos = {}
        for pos, oi in enumerate(boot_idx):
            if oi not in first_pos:
                first_pos[oi] = pos

        # Count appearances
        for i, pi in first_pos.items():
            li = labels_b[pi]
            if li == -1:
                continue
            for j, pj in first_pos.items():
                lj = labels_b[pj]
                if lj == -1:
                    continue
                denom[i, j] += 1
                if li == lj:
                    coassoc[i, j] += 1

    # Normalize
    coassoc_norm = np.zeros_like(coassoc)
    valid = denom > 0
    coassoc_norm[valid] = coassoc[valid] / denom[valid]
    np.fill_diagonal(coassoc_norm, 1.0)

    return coassoc_norm




# ================================================================
# 5. CONSENSUS DELTA CURVE 
# ================================================================

def cdf_area(coassoc, bins=100):
    """
    Compute empirical CDF and area under CDF.
    Returns:
        area,
        cdf array,
        bin_edges
    """
    # Extract upper triangle values
    flat = coassoc[np.triu_indices_from(coassoc, k=1)]

    # Histogram *frequencies* (NOT density)
    hist, bin_edges = np.histogram(flat, bins=bins, range=(0, 1), density=False)

    # Normalize to probabilities
    freq = hist / hist.sum()

    # Empirical CDF
    cdf = np.cumsum(freq)

    # Area under CDF
    dt = 1.0 / bins
    area = np.sum(cdf * dt)

    return area, cdf, bin_edges


def consensus_delta_curve(coassoc, K_range, linkage_method="average"):
    """
    Compute delta area curve for k in K_range.
    Returns:
        areas[k], deltas[k], cdfs[k]
    """
    D = 1 - coassoc
    areas = {}
    cdfs = {}
    deltas = {}

    prev_area = None

    for k in K_range:
        # 1. Consensus clustering for this k
        labels = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage=linkage_method
        ).fit_predict(D)

        # 2. Within-cluster consensus values
        mask = labels[:, None] == labels[None, :]
        cluster_only = coassoc * mask

        # 3. Compute CDF area
        area, cdf, _ = cdf_area(cluster_only)

        areas[k] = area
        cdfs[k] = cdf

        # 4. Delta = A(k) − A(k−1)
        if prev_area is None:
            deltas[k] = 0.0
        else:
            deltas[k] = area - prev_area

        prev_area = area

    return areas, deltas, cdfs


def pick_best_k(areas, deltas, method="max_delta"):
    """
    Select the best k from the delta curve.
    method:
        - "max_delta": choose k with largest increase
        - "elbow": choose elbow (second derivative)
    """
    if method == "max_delta":
        kd = list(deltas.keys())[1:]
        dv = [deltas[c] for c in kd]
        return kd[int(np.argmax(dv))]

    else:
        raise NotImplementedError("elbow method can be added later")

from kneed import KneeLocator

def elbow_kneedle(deltas):
    ks = np.array(sorted(deltas.keys()))
    dv = np.array([deltas[c] for c in ks])

    kneedle = KneeLocator(
        ks, dv,
        curve='concave',
        direction='decreasing'
    )
    return kneedle.knee


# ================================================================
# 6. CONSENSUS CLUSTERING + DENDROGRAM ORDER
# ================================================================

def consensus_clustering(coassoc, k,):
    """
    Final consensus clustering.
    """
    D = 1 - coassoc
    labels = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average"
    ).fit_predict(D)
    return labels


def dendrogram_order(coassoc, linkage_method="average"):
    """
    Compute leaf order of the dendrogram for heatmap sorting.
    """
    D = 1 - coassoc
    tri = D[np.triu_indices_from(D, k=1)]
    Z = linkage(tri, method=linkage_method)
    order = leaves_list(Z)
    return order, Z


# ================================================================
# 7. PLOTTING
# ================================================================

def plot_heatmap(mat, title, vmin=0, vmax=1):
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat, cmap="Reds", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_delta_curve(K_range, deltas):
    plt.figure(figsize=(7, 4))
    plt.plot(list(K_range), deltas, marker="o")
    plt.xlabel("k")
    plt.ylabel("Delta-CDF")
    plt.title("Consensus Delta Curve (choose elbow / max jump [chosen])")
    plt.tight_layout()
    plt.show()


def plot_dendrogram(Z):
    plt.figure(figsize=(10, 4))
    dendrogram(Z, no_labels=True)
    plt.title("Consensus Dendrogram")
    plt.tight_layout()
    plt.show()


def plot_umap_clusters(points_2d, labels, title):
    plt.figure(figsize=(9, 7))
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab20")

    for i, lab in enumerate(uniq):
        m = labels == lab
        plt.scatter(points_2d[m, 0], points_2d[m, 1],
                    s=18, alpha=0.85, label=f"cluster {lab}", c=[cmap(i % 10)])

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_consensus_with_labels(coassoc, labels, title="", cmap="Reds"):
    """
    coassoc: NxN consensus matrix
    labels: cluster labels (length N)
    """

    # 1. Sort by cluster assignments
    order = np.argsort(labels)
    sorted_mat = coassoc[np.ix_(order, order)]
    sorted_labels = labels[order]

    # 2. Create a color palette for clusters
    uniq = np.unique(sorted_labels)
    palette = sns.color_palette("tab20", len(uniq))
    label_colors = {u: palette[i] for i, u in enumerate(uniq)}

    # 3. Convert labels → list of colors in order
    row_colors = [label_colors[l] for l in sorted_labels]

    # 4. Plot heatmap with label bars
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1],
                          wspace=0.05, hspace=0.05)

    # # Top color bar (cluster labels)
    # ax_top = fig.add_subplot(gs[0, 1])
    # ax_top.imshow([row_colors], aspect='auto')
    # ax_top.axis("off")

    # # Left color bar (cluster labels)
    # ax_left = fig.add_subplot(gs[1, 0])
    # ax_left.imshow(np.array([row_colors]).T, aspect='auto')
    # ax_left.axis("off")

    # Main heatmap
    ax = fig.add_subplot(gs[1, 1])
    sns.heatmap(sorted_mat, cmap=cmap, vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Consensus"})
    ax.set_title(title)

    plt.show()

def plot_pca_variance(pca_full):
    """
    Plots the Cumulative Explained Variance
    """
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)
    ks = np.arange(1, len(evr) + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(ks, evr, alpha=0.6, label="Individual Explained Variance")
    plt.plot(ks, cum, marker='o', color='red', label="Cumulative Explained Variance")

    plt.axhline(0.95, color='green', linestyle='--', label="95% Threshold")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot (Variance Explained)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# def plot_pca_loadings(pca_model, feature_names, top_n=10):
#     """
#     Plots loadings (weights) of top PCs for interpretability.
#     """
#     loadings = pca_model.components_  # shape: (n_components, n_features)
#     n_pcs = loadings.shape[0]
#
#     for pc in range(min(n_pcs, 10)):
#         comp = loadings[pc]
#         abs_idx = np.argsort(np.abs(comp))[::-1][:top_n]
#
#         plt.figure(figsize=(10, 4))
#         plt.bar(
#             np.array(feature_names)[abs_idx],
#             comp[abs_idx],
#             color="royalblue"
#         )
#
#         plt.axhline(0, color='black')
#         plt.xticks(rotation=45, ha="right")
#         plt.title(f"PCA Loadings: PC{pc+1} (Top {top_n} features)")
#         plt.tight_layout()
#         plt.show()

def plot_pca_loadings(pca_model, feature_names, top_n=10):
    """
    Plots loadings for each PC (top_n strongest features)
    and prints the numeric loading value on the bars.
    """
    loadings = pca_model.components_
    n_pcs = loadings.shape[0]

    for pc in range(min(n_pcs, 5)):  # show first 5 PCs
        comp = loadings[pc]

        # strongest absolute loadings
        abs_idx = np.argsort(np.abs(comp))[::-1][:top_n]
        sel_features = np.array(feature_names)[abs_idx]
        sel_values = comp[abs_idx]

        plt.figure(figsize=(12, 4))
        bars = plt.bar(sel_features, sel_values, color="royalblue")

        # ---- ADD VALUE LABELS ON TOP OF BARS ----
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


# ================================================================
# 8. CLUSTER PROFILES + SIGNIFICANCE
# ================================================================

def cluster_profiles2(df_asd, features, label_col="consensus_cluster"):
    return (df_asd.groupby(label_col)[features]
            .agg(["count", "mean", "std", "median"])
            .sort_index())

def cluster_profiles(df_asd, features, label_col="consensus_cluster",
                     outdir="cluster_profiles"):
    """
    Creates per feature stats :  count, mean, std, median - for each consensus cluster.
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



def significance_tests2(df_asd, features, label_col="consensus_cluster"):
    print("\n=== Significance tests across consensus clusters ===")
    clusters = sorted(df_asd[label_col].unique())

    for f in features:
        groups = [df_asd[df_asd[label_col] == c][f].values for c in clusters]

        try:
            print("\n=== One Way F Test ===")
            F, p_a = f_oneway(*groups)
        except Exception:
            F, p_a = np.nan, np.nan

        try:
            print("\n===  Kruskal Test ===")
            H, p_k = kruskal(*groups)
        except Exception:
            H, p_k = np.nan, np.nan

        print(f"\nFeature: {f}")
        print(f"  ANOVA:   F={F:.3f}, p={p_a:.3g}")
        print(f"  Kruskal: H={H:.3f}, p={p_k:.3g}")

    print("\n=== Dunn posthoc Test for Pairwise Clusters ===")
    for f in features:
        dunn = sp.posthoc_dunn(
            df_asd, val_col=f, group_col=label_col, p_adjust="bonferroni"
        )
        print(f"\nDunn p-values for {f}:")
        print(dunn)

def significance_tests(df_asd, features, label_col="consensus_cluster",
                         _dir="significance_tests"):
    """
    significance tests
      • ANOVA + Kruskal results
      • Dunn matrices
    """


    os.makedirs(_dir, exist_ok=True)

    print("\n=== Significance tests across consensus clusters ===")
    clusters = sorted(df_asd[label_col].unique())

    summary_rows = []

    # ===============================================================
    # ANOVA + KRUSKAL
    # ===============================================================
    for f in features:
        groups = [df_asd[df_asd[label_col] == c][f].values for c in clusters]

        try:
            print("\n=== One Way F Test ===")
            F, p_a = f_oneway(*groups)
        except Exception:
            F, p_a = np.nan, np.nan

        try:
            print("\n===  Kruskal Test ===")
            H, p_k = kruskal(*groups)
        except Exception:
            H, p_k = np.nan, np.nan

        print(f"\nFeature: {f}")
        print(f"  ANOVA:   F={F:.3f}, p={p_a:.3g}")
        print(f"  Kruskal: H={H:.3f}, p={p_k:.3g}")

        summary_rows.append({
            "feature": f,
            "anova_F": F,
            "anova_p": p_a,
            "kruskal_H": H,
            "kruskal_p": p_k
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df_rounded = summary_df.round(3)

    fig, ax = plt.subplots(figsize=(12, 0.7 + 0.3 * len(summary_df_rounded)))
    ax.axis("off")

    plt.text(
        0.5, 1.01,
        f"One Way ANOVA and KRUSKAL Test Results for Consensus clusters",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold",
        transform=ax.transAxes
    )

    table = ax.table(
        cellText=summary_df_rounded.values,
        colLabels=summary_df_rounded.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    fname = os.path.join(_dir, f"significance.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"\nSaved significance test summary : {fname}")

    # ===============================================================
    # DUNN Posthoc Tset
    # ===============================================================
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

def train_asd_consensus_classifier(X_asd_embed, consensus_labels):
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42
    )
    clf.fit(X_asd_embed, consensus_labels)
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


def td_contamination(df_asd, df_td, label_col="consensus_cluster", pdf_filename="TD_Contamination"):
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
        "FSR",
        "BIS",
        "SRS.Raw",
        "TDNorm_avg_PE",
        "overall_avg_PE",
        "TDnorm_concept_learning",
        "overall_concept_learning",
        # "shortness_score",
        # "lexical_diversity",
        # "sentiment_polarity",
        # "sentiment_subjectivity",
        # "flesch_reading_ease",
        # "flesch_kincaid_grade",
    ]
    COLS = ["sub","td_or_asd"] + FEATURES
    EMBED_METHOD = "pca_umap"   # "pca" or "pca_umap"
    B_BOOT = 100
    K_RANGE = range(2, 30)

    # HDBSCAN parameters
    MIN_CLUSTER_SIZE = 10
    MIN_SAMPLES = 5

    # PCA / UMAP parameters
    PCA_VAR = 0.95
    UMAP_N_NEIGHBORS = 20
    UMAP_MIN_DIST = 0.1
    UMAP_N_COMPONENTS = 10

    RANDOM_STATE = 42
    # ---------------------------------------------
    df = pd.read_csv(data_path)
    print(f"Loaded raw data: {df.shape} from {data_path}")

    # -------- Extract NLP features --------
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

    # Embed ASD
    X_asd_embed, pca_model, umap_model = embed_data(
        X_asd_scaled,
        method=EMBED_METHOD,
        pca_var=PCA_VAR,
        umap_n_neighbors=UMAP_N_NEIGHBORS,
        umap_min_dist=UMAP_MIN_DIST,
        umap_n_components=UMAP_N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    # plot_pca_variance(pca_full)
    plot_pca_loadings(pca_model, FEATURES, top_n=10)

    # Single-run HDBSCAN
    labels_hdb, _ = cluster_hdbscan(
        X_asd_embed,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES
    )
    print("\nHDBSCAN raw clusters:", len((set(labels_hdb) - {-1})))

    # # Stability on HDBSCAN clusters
    # stab_final, _ = cluster_stability(
    #     X_asd_scaled,
    #     labels_hdb,
    #     embed_method=EMBED_METHOD,
    #     B=B_BOOT,
    #     pca_var=PCA_VAR,
    #     umap_n_neighbors=UMAP_N_NEIGHBORS,
    #     umap_min_dist=UMAP_MIN_DIST,
    #     umap_n_components=UMAP_N_COMPONENTS,
    #     min_cluster_size=MIN_CLUSTER_SIZE,
    #     min_samples=MIN_SAMPLES,
    #     random_state=RANDOM_STATE
    # )
    # print("\nPer-cluster stability:")
    # for k in sorted(stab_final):
    #     print(f"Cluster_{k}: {stab_final[k]:.4f}")

    # Co-association matrix
    coassoc = coassociation_matrix(
        X_asd_scaled,
        embed_method=EMBED_METHOD,
        B=B_BOOT,
        pca_var=PCA_VAR,
        umap_n_neighbors=UMAP_N_NEIGHBORS,
        umap_min_dist=UMAP_MIN_DIST,
        umap_n_components=UMAP_N_COMPONENTS,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        random_state=RANDOM_STATE
    )

    # Consensus delta curve -> best k
    areas, deltas, cdfs = consensus_delta_curve(coassoc, K_RANGE)
    # best_k = pick_best_k(areas, deltas)
    best_k = elbow_kneedle(deltas)
    # best_k = 12

    plt.figure(figsize=(7, 4))
    plt.plot(list(deltas.keys()), list(deltas.values()), marker="o")
    plt.xlabel("k")
    plt.ylabel("Delta Area")
    plt.title("Consensus Delta Curve")
    plt.tight_layout()
    plt.show()

    print("Best k:", best_k)

    # Consensus clustering
    consensus_labels = consensus_clustering(coassoc, 6)
    df_asd["consensus_cluster"] = consensus_labels

    print("\nConsensus cluster sizes:")
    print(df_asd["consensus_cluster"].value_counts().sort_index())

    # Dendrogram ordering + sorted heatmap
    order, Z = dendrogram_order(coassoc)
    # plot_heatmap(coassoc[np.ix_(order, order)],
    #              f"Consensus Co-association (sorted), k={best_k}")

    plot_consensus_with_labels(
        coassoc=coassoc,
        labels=consensus_labels,
        title="Consensus Co-association (sorted), k=19",
        cmap="Reds"
    )

    plot_dendrogram(Z)

    # UMAP2D visualization
    umap2d = fit_umap_2d(X_asd_scaled, random_state=RANDOM_STATE)
    plot_umap_clusters(umap2d, consensus_labels,
                       f"ASD Consensus Clusters (k={best_k})")

    # Profiles + significance
    prof = cluster_profiles(df_asd, FEATURES)
    print("\n=== Consensus cluster profiles ===")
    for p in prof:
        print("\n")
        print(p)
        print(prof[p])

    significance_tests(df_asd, FEATURES)

    # Silhouette on consensus distance (optional)
    try:
        sil = silhouette_score(1 - coassoc, consensus_labels, metric="precomputed")
        print(f"\nSilhouette (consensus distance): {sil:.3f}")
    except Exception:
        print("\nSilhouette failed (degenerate distances likely).")

    # TD projection + contamination
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
            f"TD Contamination Results:",
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

        fname = ("TD_Contamination_Report.pdf")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

    # Save ASD-labeled output
    df_asd.to_csv("asd_consensus_labeled.csv", index=False)
    print("\nSaved ASD consensus labels: asd_consensus_labeled.csv")

    # Consensus clustering
    for temp_k in range(2,best_k+1):
        consensus_labels = consensus_clustering(coassoc, temp_k)
        sil = silhouette_score(1 - coassoc, consensus_labels, metric="precomputed")
        print(f"\nSilhouette (k = {temp_k}): {sil:.3f}")

        plot_consensus_with_labels(
            coassoc=coassoc,
            labels=consensus_labels,
            title=f"Consensus Co-association (sorted), k={temp_k}",
            cmap="Reds"
        )