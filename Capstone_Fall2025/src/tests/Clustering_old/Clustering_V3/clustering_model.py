# clustering_model.py (V3)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
import hdbscan
import json
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessor import preprocess_clustering_data
from visualization import (
    plot_correlation_heatmap,
    plot_tsne_projection,
    plot_tsne_projection_streamlit,
    plot_hdbscan_probability,
    plot_cluster_feature_means,
    plot_feature_variance,
    plot_tsne_cluster_explorer,
)


# ------------------------------------------------------------
data_version = "Data_Clustering_v3"
model_version = "Clustering_V3"
# ------------------------------------------------------------


class ClusterAnalyzerV3:
    """HDBSCAN clustering on t-SNE embeddings (nonlinear structure discovery)."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.results_dir = self.project_root / "Results" / model_version / "visualizations"
        self.td_asd_dir = self.results_dir / "td_asd_clusters"
        self.asd_dir = self.results_dir / "asd_subclusters"
        self.td_asd_dir.mkdir(parents=True, exist_ok=True)
        self.asd_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    def run_pipeline(self, df):
        """Executes full t-SNE + HDBSCAN analysis."""
        print("\n[1] Correlation Heatmap...")
        plot_correlation_heatmap(df, self.td_asd_dir / "correlation_matrix.png")

        print("[2] t-SNE + HDBSCAN (TD + ASD)...")
        self.run_tsne_hdbscan(df, save_dir=self.td_asd_dir)

        print("[3] ASD-only Subclustering (t-SNE + HDBSCAN)...")
        asd_df = df[df["td_or_asd"] == 1].reset_index(drop=True)
        self.run_tsne_hdbscan(asd_df, save_dir=self.asd_dir)

        print("\n✅ Clustering_V3 pipeline completed successfully.\n")

    # --------------------------------------------------------
    def run_tsne_hdbscan(self, df, save_dir, method_name="hdbscan"):
        """Runs t-SNE embedding + HDBSCAN clustering."""
        features = [
            'FSR_scaled', 'BIS_scaled', 'SRS.Raw_scaled',
            'TDNorm_avg_PE_scaled', 'overall_avg_PE_scaled',
            'TDnorm_concept_learning_scaled', 'overall_concept_learning_scaled'
        ]
        target_col = "td_or_asd"
        X = df[features].values

        # === Subfolder structure ===
        proj_dir = save_dir / "projections"
        feat_dir = save_dir / "feature_visuals"
        for d in [proj_dir, feat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # === t-SNE Embedding ===
        print("   → Running t-SNE embedding (may take several minutes)...")
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            max_iter=2000,
            random_state=42,
            verbose=1
        )
        X_tsne = tsne.fit_transform(X)

        # === HDBSCAN Clustering ===
        print("   → Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        labels = clusterer.fit_predict(X_tsne)
        df["cluster"] = labels

        # === Summary ===
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters[unique_clusters != -1])
        n_noise = np.sum(labels == -1)
        print(f"   ✓ Found {n_clusters} clusters (+ {n_noise} noise points)")

        # === Cluster Distribution Visualization ===
        print("   → Plotting cluster distribution...")
        plt.figure(figsize=(7, 4))
        cluster_counts = df["cluster"].value_counts().sort_index()
        colors = ["#999999" if idx == -1 else sns.color_palette("tab10")[i % 10]
                  for i, idx in enumerate(cluster_counts.index)]
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=colors)
        plt.title("HDBSCAN Cluster Distribution (including Noise)", fontsize=13, fontweight="bold")
        plt.xlabel("Cluster Label (-1 = Noise)")
        plt.ylabel("Number of Points")
        plt.tight_layout()
        plt.savefig(proj_dir / "hdbscan_cluster_distribution.png", dpi=300)
        plt.savefig(proj_dir / "hdbscan_cluster_distribution.pdf", dpi=300)
        plt.close()

        # === Visualization: t-SNE projections ===
        # Clusters
        plot_tsne_projection(
            X_tsne, df, "cluster",
            proj_dir / f"tsne_hdbscan_clusters.png",
            title="HDBSCAN Clusters (2D t-SNE Projection)"
        )

        # Target
        if target_col in df.columns:
            plot_tsne_projection(
                X_tsne, df, target_col,
                proj_dir / f"tsne_target_projection.png",
                title="TD/ASD Participants (t-SNE Projection)"
            )

        # === HDBSCAN Probability Visualization ===
        plot_hdbscan_probability(
            X_tsne, clusterer, proj_dir / "hdbscan_probability_map.png"
        )

        # === Streamlit Interactive Versions ===
        plot_tsne_projection_streamlit(
            X_tsne, df, color_col="cluster",
            save_path=proj_dir / f"tsne_hdbscan_clusters.html"
        )
        if target_col in df.columns:
            plot_tsne_projection_streamlit(
                X_tsne, df, color_col=target_col,
                save_path=proj_dir / f"tsne_target_projection.html"
            )

        # === Feature Summaries ===
        plot_cluster_feature_means(df, features, feat_dir / "cluster_feature_means.png", method_name)
        plot_feature_variance(df, features, feat_dir / "feature_variance.png", method_name)

        # === Save Metrics Summary ===
        metrics = {
            "method": method_name,
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "unique_labels": unique_clusters.tolist(),
            "tsne_params": {
                "perplexity": 30,
                "learning_rate": 200,
                "n_iter": 2000
            },
            "hdbscan_params": {
                "min_cluster_size": 15,
                "min_samples": 5
            },
            "cluster_counts": cluster_counts.to_dict()
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics + visuals saved in {save_dir}")
        # ------------------ SAVE FINAL PROCESSED CSV ------------------
        df_final = df.copy()

        # Add embeddings
        if X_tsne is not None:
            df_final["tSNE1"] = X_tsne[:, 0]
            df_final["tSNE2"] = X_tsne[:, 1]

        # Add cluster column
        df_final["cluster"] = labels

        # Save full dataset with everything
        output_csv = save_dir / "processed_with_clusters.csv"
        df_final.to_csv(output_csv, index=False)

        print(f"Saved full processed dataset → {output_csv}")


        # === Interactive cluster explorer with stats ===
        numeric_cols = [
            'FSR', 'BIS', 'SRS.Raw',
            'TDNorm_avg_PE', 'overall_avg_PE',
            'TDnorm_concept_learning', 'overall_concept_learning'
        ]
        plot_tsne_cluster_explorer(
            X_tsne,
            df,
            numeric_cols=numeric_cols,
            save_path=proj_dir / f"tsne_cluster_explorer_{method_name}.html",
        )
        print("   → Interactive cluster explorer saved.")


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print("\n=== Running Clustering_V3 Pipeline ===\n")
    df_clean = preprocess_clustering_data(project_root)
    analyzer = ClusterAnalyzerV3(project_root)
    analyzer.run_pipeline(df_clean)
