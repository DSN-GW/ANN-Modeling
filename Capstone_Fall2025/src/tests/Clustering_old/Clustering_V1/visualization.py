import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json


# ===================== HELPER =====================

def save_dual_format(save_path):
    """Saves the current figure in both .png and .pdf formats."""
    save_path = Path(save_path)
    for ext in ["png", "pdf"]:
        plt.savefig(save_path.with_suffix(f".{ext}"), dpi=300, bbox_inches='tight')
    plt.close()


# ===================== BASIC VISUALS =====================

def plot_correlation_heatmap(df, save_path):
    """Plots correlation matrix for numeric columns only."""
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix (Numeric Columns Only)")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_pca_projection(X_pca, df, color_col, save_path, title=None):
    """2D PCA projection for participants or KMeans clusters with consistent colors."""
    plt.figure(figsize=(8, 6))

    # === Color & title logic (mirrors V3 t-SNE) ===
    if color_col == "td_or_asd":
        palette = {0: "#9FE2BF", 1: "#FF9999"}  # TD=green, ASD=red (soft)
        unique_vals = set(df[color_col].unique())
        if unique_vals == {0, 1}:
            plot_title = "2D Representation of TD/ASD Participants"
        elif unique_vals == {1}:
            plot_title = "2D Representation of ASD Participants"
        elif unique_vals == {0}:
            plot_title = "2D Representation of TD Participants"
        else:
            plot_title = "2D Representation of Participants"
        final_title = plot_title
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        palette = sns.color_palette("tab20", n_clusters)  # consistent with V3
        plot_title = "2D Representation of KMeans Clusters"
        final_title = title or plot_title
    else:
        palette = "husl"
        plot_title = f"2D Representation of {color_col}"
        final_title = title or plot_title

    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=df[color_col],
        palette=palette,
        s=70, alpha=0.9,
        edgecolor="black", linewidth=0.4
    )

    plt.title(final_title, fontsize=13, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(
        title=color_col,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True, facecolor='white', framealpha=0.6
    )
    plt.tight_layout(pad=1.2)
    save_dual_format(save_path)


# ===================== EVALUATION METRICS =====================

def plot_silhouette_plot(k_values, scores, save_path, title="Silhouette Score vs K"):
    """Silhouette score vs number of clusters."""
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker='o', color='teal')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_db_index_vs_k(k_values, scores, save_path, title="Davies–Bouldin Index vs K"):
    """Davies–Bouldin Index vs number of clusters."""
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker='o', color='indianred')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies–Bouldin Index")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_ari_nmi_bar(k_values, ari_scores, nmi_scores, save_path):
    """Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) comparison."""
    x = np.arange(len(k_values))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, ari_scores, width, label='ARI', color='steelblue')
    plt.bar(x + width / 2, nmi_scores, width, label='NMI', color='salmon')
    plt.xticks(x, k_values)
    plt.title("ARI vs NMI Across Cluster Sizes")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    save_dual_format(save_path)


# ===================== FEATURE IMPORTANCE VISUALS =====================

def plot_pca_feature_loadings(pca, features, save_path, method_name="kmeans"):
    """Visualizes PCA feature contributions (component loadings)."""
    plt.figure(figsize=(8, 5))
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=features
    )
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title(f"PCA Feature Contributions ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_dual_format(save_path)


def plot_cluster_feature_means(df, features, save_path, method_name="kmeans"):
    """Mean feature values per cluster."""
    cluster_means = df.groupby("cluster")[features].mean()
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm")
    plt.title(f"Cluster Centroids by Feature ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_dual_format(save_path)


def plot_feature_variance(df, features, save_path, method_name="kmeans"):
    """Feature variance across clusters."""
    feature_variance = df.groupby("cluster")[features].mean().var()
    plt.figure(figsize=(8, 4))
    feature_variance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title(f"Feature Variance Across Clusters ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Variance of Cluster Means")
    plt.tight_layout()
    save_dual_format(save_path)


# ===================== STREAMLIT VISUALS =====================

def plot_pca_projection_streamlit(X_pca, df, color_col, save_path, title=None):
    """
    Generates interactive PCA projection for Streamlit and saves as .html.
    Hover shows only: td_or_asd, cluster, PC1, PC2.
    """
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    # === Color and title logic (mirrors V3) ===
    if color_col == "td_or_asd":
        color_map = {0: "#9FE2BF", 1: "#FF9999"}  # TD = soft green, ASD = soft red
        if df_plot[color_col].nunique() == 1:
            plot_title = "2D Representation of ASD Participants"
        else:
            plot_title = "2D Representation of TD/ASD Participants"
    elif color_col == "cluster":
        color_map = None
        plot_title = "2D Representation of KMeans Clusters"
    else:
        color_map = None
        plot_title = title or f"2D Representation of {color_col}"

    fig = px.scatter(
        df_plot,
        x="PC1", y="PC2",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={
            "td_or_asd": True if "td_or_asd" in df_plot.columns else False,
            "cluster": True if "cluster" in df_plot.columns else False,
            "PC1": ':.3f',
            "PC2": ':.3f'
        },
        title=plot_title,
        template="plotly_white"
    )

    fig.update_layout(
        height=600,
        title_font=dict(size=16),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title=color_col,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs='inline')
    print(f"✅ Streamlit-ready interactive PCA saved: {html_path}")


# ===================== CLUSTER EXPLORER (V1-style, like V3) =====================

def _get_cluster_stats_text(df, numeric_cols, cluster_id):
    """
    Build an HTML snippet summarizing one cluster:
    - TD / ASD counts
    - min / max / mean for selected numeric columns
    """
    sub = df[df["cluster"] == cluster_id]

    if len(sub) == 0:
        return f"<b>Cluster {cluster_id}</b><br>No points in this cluster."

    td_count = int((sub["td_or_asd"] == 0).sum()) if "td_or_asd" in sub.columns else 0
    asd_count = int((sub["td_or_asd"] == 1).sum()) if "td_or_asd" in sub.columns else 0

    lines = []
    lines.append(f"<b>Cluster {cluster_id}</b>")
    lines.append(f"Total: {len(sub)}")
    if "td_or_asd" in sub.columns:
        lines.append(f"TD: {td_count}")
        lines.append(f"ASD: {asd_count}")
    lines.append("")

    for col in numeric_cols:
        if col not in sub.columns:
            continue
        col_min = sub[col].min()
        col_max = sub[col].max()
        col_mean = sub[col].mean()
        lines.append(f"<b>{col}</b>")
        lines.append(f"min = {col_min:.3f}")
        lines.append(f"max = {col_max:.3f}")
        lines.append(f"mean = {col_mean:.3f}")
        lines.append("")

    return "<br>".join(lines)


def plot_pca_cluster_explorer(X_pca, df, numeric_cols, save_path):
    """
    Interactive PCA cluster explorer for V1:
    - Scatter by PC1 / PC2
    - Each cluster is a separate trace
    - Right-side panel shows descriptive stats for the selected cluster
    - Clicking a cluster in the legend updates the stats panel
    """
    save_path = Path(save_path)

    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    clusters = sorted(df_plot["cluster"].unique())

    # --- color palette: tab20 + grey for noise (-1), like V3 ---
    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgba(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgba({r},{g},{b},1)"

    fig = go.Figure()

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]

        if c == -1:
            color = noise_color
        else:
            color = _to_rgba(base_palette[i % len(base_palette)])

        fig.add_trace(
            go.Scatter(
                x=sub["PC1"],
                y=sub["PC2"],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(
                    size=7,
                    color=color,
                    line=dict(width=0.8, color="black"),  # border around each point
                ),
                hovertemplate=(
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    f"Cluster: {c}<extra></extra>"
                ),
            )
        )

    # --- stats text for each cluster ---
    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_cluster = clusters[0]
    first_key = str(first_cluster)

    fig.update_layout(
        title="2D Representation of KMeans Clusters",
        xaxis=dict(title="Principal Component 1", domain=[0.0, 0.60]),
        yaxis=dict(title="Principal Component 2"),
        hovermode="closest",
        width=1200,
        height=700,
        margin=dict(l=60, r=200, t=60, b=60),
        legend=dict(
            title="Clusters",
            x=0.62,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
    )

    # === JS: external scrollable + collapsible stats panel (mirrors V3) ===
    js_stats = json.dumps(stats_text_map)

    post_script = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        // ---- create side panel container ----
        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel';

        var rect = plot.getBoundingClientRect();
        panel.style.position = 'absolute';
        panel.style.top = (rect.top + window.scrollY + 40) + 'px';
        panel.style.left = (rect.left + window.scrollX + rect.width * 0.75) + 'px';
        panel.style.width = '260px';
        panel.style.maxHeight = '500px';
        panel.style.overflowY = 'auto';
        panel.style.border = '1px solid #333';
        panel.style.background = 'white';
        panel.style.padding = '8px 10px';
        panel.style.boxShadow = '0 0 4px rgba(0,0,0,0.2)';
        panel.style.fontFamily = 'Arial, sans-serif';
        panel.style.fontSize = '12px';
        panel.style.boxSizing = 'border-box';
        panel.style.zIndex = 1000;

        var header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
        header.style.marginBottom = '6px';

        var title = document.createElement('span');
        title.textContent = 'Cluster stats';

        var btn = document.createElement('button');
        btn.textContent = '\\u2212';
        btn.style.marginLeft = '8px';
        btn.style.cursor = 'pointer';
        btn.style.border = '1px solid #888';
        btn.style.background = '#f7f7f7';
        btn.style.padding = '0 6px';

        header.appendChild(title);
        header.appendChild(btn);

        var content = document.createElement('div');
        content.id = 'cluster-stats-content';
        content.innerHTML = statsByCluster['{first_key}'] || 'No stats available';

        panel.appendChild(header);
        panel.appendChild(content);
        document.body.appendChild(panel);

        var collapsed = false;
        btn.addEventListener('click', function() {{
            collapsed = !collapsed;
            if (collapsed) {{
                content.style.display = 'none';
                btn.textContent = '+';
            }} else {{
                content.style.display = 'block';
                btn.textContent = '\\u2212';
            }}
        }});

        // ---- update stats when legend item is clicked ----
        plot.on('plotly_legendclick', function(event) {{
            var trace = event.data[event.curveNumber];
            var name = trace.name || "";
            var cid = name.replace("Cluster ", "");
            var text = statsByCluster[cid];
            if (text) {{
                content.innerHTML = text;
            }}
        }});
    }});
    </script>
    """

    # ---- generate HTML and inject JS before </body> ----
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    insert_pos = html_str.rfind("</body>")
    if insert_pos != -1:
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive PCA cluster explorer saved → {save_path}")
