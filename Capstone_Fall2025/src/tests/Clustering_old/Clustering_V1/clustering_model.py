import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
import json
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessor import preprocess_clustering_data
from visualization import (
    plot_correlation_heatmap,
    plot_pca_projection,
    plot_pca_projection_streamlit,
    plot_silhouette_plot,
    plot_db_index_vs_k,
    plot_ari_nmi_bar,
    plot_pca_feature_loadings,
    plot_cluster_feature_means,
    plot_feature_variance,
    plot_pca_cluster_explorer,  # NEW: interactive explorer
)

# ---------------------------------------------------------
data_version = "Data_Clustering_v1"
model_version = "Clustering_V1"
# ---------------------------------------------------------


class ClusterAnalyzer:
    """Complete unsupervised clustering pipeline using PCA + KMeans."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.results_dir = self.project_root / "Results" / model_version / "visualizations"
        self.td_asd_dir = self.results_dir / "td_asd_clusters"
        self.asd_dir = self.results_dir / "asd_subclusters"
        self.td_asd_dir.mkdir(parents=True, exist_ok=True)
        self.asd_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    def run_pipeline(self, df):
        print("\n[1] Correlation Heatmap...")
        plot_correlation_heatmap(df, self.td_asd_dir / "correlation_matrix.png")

        print("[2] PCA + KMeans (TD + ASD)...")
        self.run_pca_kmeans(df, save_dir=self.td_asd_dir, cluster_range=(2, 6))

        print("[3] ASD-only Subclustering...")
        asd_df = df[df["td_or_asd"] == 1].reset_index(drop=True)
        self.run_pca_kmeans(asd_df, save_dir=self.asd_dir, cluster_range=(2, 6))

        print("\n✅ Clustering_V1 pipeline completed successfully.\n")

    # ---------------------------------------------------------
    def run_pca_kmeans(self, df, save_dir, cluster_range=(2, 6), method_name="kmeans"):
        """Performs PCA + KMeans clustering and saves all results."""
        features = [
            'FSR_scaled', 'BIS_scaled', 'SRS.Raw_scaled',
            'TDNorm_avg_PE_scaled', 'overall_avg_PE_scaled',
            'TDnorm_concept_learning_scaled', 'overall_concept_learning_scaled'
        ]
        print("\n📊 Selected Features for PCA + KMeans:")
        for f in features:
            print(f"   • {f}")
        print(f"Total features used: {len(features)}")

        target_col = "td_or_asd"
        X = df[features].values

        # === Subfolder structure ===
        proj_dir = save_dir / "projections"
        metric_dir = save_dir / "metrics_visuals"
        feat_dir = save_dir / "feature_visuals"
        for d in [proj_dir, metric_dir, feat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # === PCA Transformation ===
        print("   → Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        # === PCA Variance Plot ===
        plt.figure(figsize=(6, 4))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.title("Explained Variance by PCA Components")
        plt.xlabel("Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.tight_layout()
        plt.savefig(proj_dir / "pca_variance.png", dpi=300)
        plt.savefig(proj_dir / "pca_variance.pdf", dpi=300)
        plt.close()

        # === Evaluate KMeans over range ===
        silhouette_scores, db_scores, ari_scores, nmi_scores = [], [], [], []

        print("   → Evaluating KMeans over k range...")
        k_values = list(range(cluster_range[0], cluster_range[1] + 1))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X_pca)
            silhouette_scores.append(silhouette_score(X_pca, labels))
            db_scores.append(davies_bouldin_score(X_pca, labels))
            if target_col in df.columns:
                ari_scores.append(adjusted_rand_score(df[target_col], labels))
                nmi_scores.append(normalized_mutual_info_score(df[target_col], labels))

        # === Metric plots ===
        plot_silhouette_plot(
            k_values, silhouette_scores,
            metric_dir / f"silhouette_plot_{method_name}.png",
            title=f"Silhouette Score vs K ({method_name.upper()})"
        )

        plot_db_index_vs_k(
            k_values, db_scores,
            metric_dir / f"db_index_vs_k_{method_name}.png",
            title=f"Davies–Bouldin Index vs K ({method_name.upper()})"
        )

        if ari_scores and nmi_scores:
            plot_ari_nmi_bar(k_values, ari_scores, nmi_scores, metric_dir / "ari_nmi_bar.png")

        # === Pick best k ===
        best_k = k_values[np.argmax(silhouette_scores)]
        print(f"   ✓ Best number of clusters (Silhouette): {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_pca)
        df["cluster"] = labels

        # === Cluster Distribution Visualization (like V3) ===
        print("   → Plotting cluster distribution...")
        plt.figure(figsize=(7, 4))
        cluster_counts = df["cluster"].value_counts().sort_index()
        colors = [sns.color_palette("tab10")[i % 10] for i, _ in enumerate(cluster_counts.index)]
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=colors)
        plt.title("KMeans Cluster Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Cluster Label")
        plt.ylabel("Number of Points")
        plt.tight_layout()
        plt.savefig(proj_dir / "kmeans_cluster_distribution.png", dpi=300)
        plt.savefig(proj_dir / "kmeans_cluster_distribution.pdf", dpi=300)
        plt.close()

        # === Feature Importance Visualizations ===
        plot_pca_feature_loadings(
            pca, features,
            feat_dir / f"pca_feature_loadings_{method_name}.png",
            method_name
        )
        plot_cluster_feature_means(
            df, features,
            feat_dir / f"cluster_feature_means_{method_name}.png",
            method_name
        )
        plot_feature_variance(
            df, features,
            feat_dir / f"feature_variance_{method_name}.png",
            method_name
        )

        # === PCA Projections (static) ===
        if target_col in df.columns:
            plot_pca_projection(
                X_pca, df, target_col,
                proj_dir / "pca_target_projection.png"
            )

        plot_pca_projection(
            X_pca, df, "cluster",
            proj_dir / f"pca_cluster_projection_{method_name}.png"
        )

        # === Streamlit Interactive Plots ===
        plot_pca_projection_streamlit(
            X_pca, df, color_col="cluster",
            save_path=proj_dir / f"pca_cluster_projection_{method_name}.html"
        )
        if target_col in df.columns:
            plot_pca_projection_streamlit(
                X_pca, df, color_col=target_col,
                save_path=proj_dir / f"pca_target_projection_{method_name}.html"
            )

        # ------------------ SAVE FINAL PROCESSED CSV ------------------
        print("   → Saving processed dataset with PCA + clusters...")
        df_final = df.copy()
        if X_pca is not None:
            df_final["PC1"] = X_pca[:, 0]
            df_final["PC2"] = X_pca[:, 1]
        df_final["cluster"] = labels

        output_csv = save_dir / "processed_with_clusters.csv"
        df_final.to_csv(output_csv, index=False)
        print(f"      Saved full processed dataset → {output_csv}")

        # === Interactive cluster explorer with stats (V1 analog of V3) ===
        numeric_cols = [
            'FSR', 'BIS', 'SRS.Raw',
            'TDNorm_avg_PE', 'overall_avg_PE',
            'TDnorm_concept_learning', 'overall_concept_learning'
        ]
        print("   → Building interactive PCA cluster explorer...")
        plot_pca_cluster_explorer(
            X_pca,
            df,
            numeric_cols=numeric_cols,
            save_path=proj_dir / f"pca_cluster_explorer_{method_name}.html",
        )
        print("      Interactive cluster explorer saved.")

        # === Save Metrics Summary ===
        metrics = {
            "method": method_name,
            "data_version": data_version,
            "k_range": k_values,
            "silhouette_scores": silhouette_scores,
            "db_index_scores": db_scores,
            "ari_scores": ari_scores,
            "nmi_scores": nmi_scores,
            "best_k": int(best_k),
            "pca_params": {
                "n_components": 2,
                "random_state": 42,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
            "kmeans_params": {
                "n_clusters": int(best_k),
                "random_state": 42,
            },
            "cluster_counts": cluster_counts.to_dict(),
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics + visuals saved in {save_dir}")


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print("\n=== Running Clustering_V1 Pipeline ===\n")
    df_clean = preprocess_clustering_data(project_root)
    analyzer = ClusterAnalyzer(project_root)
    analyzer.run_pipeline(df_clean)
