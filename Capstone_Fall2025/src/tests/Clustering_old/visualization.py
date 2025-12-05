import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D projections


# ===================== HELPER =====================

def save_dual_format(save_path):
    """
    Saves the current figure in both .png and .pdf formats.
    Example: input '.../figure.png' → saves 'figure.png' and 'figure.pdf'
    """
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


def _get_td_asd_palette_and_title(df, col_name="td_or_asd"):
    """Shared palette + title logic for TD/ASD."""
    palette = {0: "#9FE2BF", 1: "#FF9999"}  # TD=green, ASD=red (soft)
    unique_vals = set(df[col_name].dropna().unique())
    if unique_vals == {0, 1}:
        title = "2D Representation of TD/ASD Participants"
    elif unique_vals == {1}:
        title = "2D Representation of ASD Participants"
    elif unique_vals == {0}:
        title = "2D Representation of TD Participants"
    else:
        title = "2D Representation of Participants"
    return palette, title


# ===================== PCA 2D PROJECTIONS =====================

def plot_pca_projection(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    2D PCA projection for participants or clusters with consistent colors.

    method_name:
        - "pca_kmeans"  → title mentions KMeans
        - "pca_gmm"     → title mentions GMM
        - None / other  → generic wording
    """
    plt.figure(figsize=(8, 6))

    # === Color & title logic ===
    if color_col == "td_or_asd":
        palette, plot_title = _get_td_asd_palette_and_title(df, "td_or_asd")
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        palette = sns.color_palette("tab20", n_clusters)  # consistent style
        if method_name == "pca_kmeans":
            plot_title = "2D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "2D Representation of GMM Clusters"
        else:
            plot_title = "2D Representation of Clusters"
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


# ===================== PCA 3D PROJECTIONS (STATIC) =====================

def plot_pca_projection_3d(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    3D PCA projection (PC1, PC2, PC3) with the same color logic as 2D.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if color_col == "td_or_asd":
        palette, plot_title = _get_td_asd_palette_and_title(df, "td_or_asd")
        colors = df[color_col].map(palette)
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        base_palette = sns.color_palette("tab20", n_clusters)
        color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(sorted(df[color_col].unique()))}
        colors = df[color_col].map(color_map)
        if method_name == "pca_kmeans":
            plot_title = "3D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "3D Representation of GMM Clusters"
        else:
            plot_title = "3D Representation of Clusters"
    else:
        plot_title = f"3D Representation of {color_col}"
        pal = sns.color_palette("husl", df[color_col].nunique())
        color_map = {c: pal[i % len(pal)] for i, c in enumerate(sorted(df[color_col].unique()))}
        colors = df[color_col].map(color_map)

    final_title = title or plot_title

    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=colors,
        s=45,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

    ax.set_title(final_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    plt.tight_layout()
    save_dual_format(save_path)


# ===================== KMEANS / PCA METRICS =====================

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


# ===================== GMM METRIC VISUALS =====================

def plot_aic_bic_vs_k(k_values, aic_scores, bic_scores, save_path, method_name="gmm"):
    """Plots AIC and BIC across different cluster counts."""
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, aic_scores, marker='o', label="AIC", color="steelblue")
    plt.plot(k_values, bic_scores, marker='s', label="BIC", color="indianred")
    plt.title(f"AIC / BIC vs K ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score (Lower is Better)")
    plt.legend()
    plt.tight_layout()
    save_dual_format(save_path)


def plot_gmm_uncertainty(X_pca, responsibilities, save_path):
    """
    Visualizes GMM cluster uncertainty:
    uncertainty = 1 - max responsibility over components.
    """
    uncertainty = 1 - np.max(responsibilities, axis=1)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=uncertainty, cmap="magma", s=50, edgecolor='k', alpha=0.8
    )
    plt.colorbar(scatter, label="Cluster Uncertainty (1 - max responsibility)")
    plt.title("GMM Cluster Uncertainty", fontsize=13, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
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
    # For HDBSCAN, ignore noise (-1) when computing centroids
    if method_name == "tsne_hdbscan":
        cluster_means = df[df["cluster"] != -1].groupby("cluster")[features].mean()
    else:
        cluster_means = df.groupby("cluster")[features].mean()

    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm")
    plt.title(f"Cluster Centroids by Feature ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_dual_format(save_path)


def plot_feature_variance(df, features, save_path, method_name="kmeans"):
    """Feature variance across clusters."""
    if method_name == "tsne_hdbscan":
        feature_variance = df[df["cluster"] != -1].groupby("cluster")[features].mean().var()
    else:
        feature_variance = df.groupby("cluster")[features].mean().var()

    plt.figure(figsize=(8, 4))
    feature_variance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title(f"Feature Variance Across Clusters ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Variance of Cluster Means")
    plt.tight_layout()
    save_dual_format(save_path)


# ===================== STREAMLIT PCA VISUALS (2D) =====================

def plot_pca_projection_streamlit(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    Interactive PCA projection for Streamlit.
    - If color_col == 'td_or_asd': use light green/red and TD/ASD legend labels.
    - Otherwise behaves as before.
    """
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    # ---------- color + title logic ----------
    color_map = None
    legend_title = color_col
    color_field = color_col  # what we actually give to px.scatter

    if color_col == "td_or_asd":
        # Map 0/1 → TD/ASD labels for the legend
        label_map = {0: "TD", 1: "ASD"}
        df_plot["td_asd_label"] = df_plot["td_or_asd"].map(label_map).fillna("Unknown")
        color_field = "td_asd_label"
        legend_title = "Group"
        color_map = {
            "TD": "#9FE2BF",   # soft green
            "ASD": "#FF9999",  # soft red
            "Unknown": "#AAAAAA",
        }

        # Title logic like the static plot
        uniq = set(df_plot["td_or_asd"].dropna().unique())
        if uniq == {0, 1}:
            plot_title = "2D Representation of TD/ASD Participants"
        elif uniq == {1}:
            plot_title = "2D Representation of ASD Participants"
        elif uniq == {0}:
            plot_title = "2D Representation of TD Participants"
        else:
            plot_title = "2D Representation of Participants"

    elif color_col == "cluster":
        color_map = None   # let Plotly pick tab20-like colors
        legend_title = "Cluster"
        if method_name == "pca_kmeans":
            plot_title = "2D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "2D Representation of GMM Clusters"
        else:
            plot_title = "2D Representation of Clusters"
    else:
        color_map = None
        plot_title = title or f"2D Representation of {color_col}"

    # ---------- figure ----------
    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color=color_field,
        color_discrete_map=color_map,
        hover_data={
            "td_or_asd": "td_or_asd" in df_plot.columns,
            "cluster": "cluster" in df_plot.columns,
            "PC1": ':.3f',
            "PC2": ':.3f',
        },
        title=plot_title,
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(
            size=7,
            line=dict(width=0.8, color="black")  # black border
        )
    )
    fig.update_layout(
        height=600,
        title_font=dict(size=16),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title=legend_title,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ Streamlit-ready interactive PCA saved: {html_path}")


# ===================== STREAMLIT PCA VISUALS (3D) =====================

def plot_pca_projection_streamlit_3d(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    Interactive 3D PCA projection for Streamlit (PC1, PC2, PC3).
    Uses same color logic as 2D.
    """
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]
    df_plot["PC3"] = X_pca[:, 2]

    color_map = None
    legend_title = color_col
    color_field = color_col

    if color_col == "td_or_asd":
        label_map = {0: "TD", 1: "ASD"}
        df_plot["td_asd_label"] = df_plot["td_or_asd"].map(label_map).fillna("Unknown")
        color_field = "td_asd_label"
        legend_title = "Group"
        color_map = {
            "TD": "#9FE2BF",
            "ASD": "#FF9999",
            "Unknown": "#AAAAAA",
        }

        _, plot_title = _get_td_asd_palette_and_title(df_plot, "td_or_asd")
        plot_title = plot_title.replace("2D", "3D")
    elif color_col == "cluster":
        legend_title = "Cluster"
        if method_name == "pca_kmeans":
            plot_title = "3D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "3D Representation of GMM Clusters"
        else:
            plot_title = "3D Representation of Clusters"
    else:
        plot_title = title or f"3D Representation of {color_col}"

    fig = px.scatter_3d(
        df_plot,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_field,
        color_discrete_map=color_map,
        hover_data={
            "cluster": "cluster" in df_plot.columns,
            "td_or_asd": "td_or_asd" in df_plot.columns,
            "PC1": ':.3f',
            "PC2": ':.3f',
            "PC3": ':.3f',
        },
        title=plot_title,
        template="plotly_white",
    )

    fig.update_traces(
        marker=dict(
            size=5,
            line=dict(width=0.7, color="black")
        )
    )
    fig.update_layout(
        height=650,
        title_font=dict(size=16),
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3",
        ),
        legend_title=legend_title,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ Streamlit-ready interactive 3D PCA saved: {html_path}")


# ===================== PCA FEATURE CONTRIBUTIONS (STREAMLIT BARS) =====================

def plot_pca_feature_contributions_streamlit(
    pca,
    feature_names,
    save_path,
    n_components=3,
    top_n=None,
):
    """
    Build a Streamlit-ready HTML with grouped bar charts:
    PC1, PC2, PC3 (or fewer if n_components < 3).
    Each bar = |loading| for a feature on that PC.

    If top_n is given, we keep only the top_n features with the
    highest max |loading| across the selected PCs.
    """
    save_path = Path(save_path)

    comps = pca.components_
    n_avail = comps.shape[0]
    n_components = min(n_components, n_avail)

    loadings = np.abs(comps[:n_components, :])  # (n_components, n_features)
    feature_names = list(feature_names)

    # overall importance across selected PCs
    max_loading = loadings.max(axis=0)  # (n_features,)
    order = np.argsort(-max_loading)

    if top_n is not None and top_n < len(feature_names):
        order = order[:top_n]

    loadings_sel = loadings[:, order]
    features_sel = [feature_names[i] for i in order]

    data = []
    for pc_idx in range(n_components):
        for j, feat in enumerate(features_sel):
            data.append({
                "Feature": feat,
                "PC": f"PC{pc_idx + 1}",
                "AbsLoading": loadings_sel[pc_idx, j],
            })

    df_plot = pd.DataFrame(data)

    fig = px.bar(
        df_plot,
        x="Feature",
        y="AbsLoading",
        color="PC",
        barmode="group",
        title="Absolute PCA Loadings (Top Features)",
        template="plotly_white",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        title_font=dict(size=16),
        yaxis_title="|Loading|",
    )

    html_path = save_path.with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ PCA feature contribution bars saved: {html_path}")


# ===================== SHARED CLUSTER-STATS HELPER =====================

def _get_cluster_stats_text(df, numeric_cols, cluster_id):
    """
    Build an HTML snippet summarizing one cluster:
    - TD / ASD counts
    - min / max / mean for selected numeric columns
    """
    sub = df[df["cluster"] == cluster_id]

    if len(sub) == 0:
        return f"<b>Cluster {cluster_id}</b><br>No points in this cluster."

    td_count = int((sub.get("td_or_asd") == 0).sum()) if "td_or_asd" in sub.columns else 0
    asd_count = int((sub.get("td_or_asd") == 1).sum()) if "td_or_asd" in sub.columns else 0

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


# ===================== PCA CLUSTER EXPLORER (KMEANS/GMM, 2D) =====================

def plot_pca_cluster_explorer(X_pca, df, numeric_cols, save_path, method_name="kmeans"):
    """
    Interactive PCA cluster explorer:
    - Scatter by PC1 / PC2
    - Each cluster is a separate trace
    - Fixed top-right stats panel (Streamlit-friendly)

    method_name controls the title wording (KMeans vs GMM).
    """
    save_path = Path(save_path)

    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    clusters = sorted(df_plot["cluster"].unique())

    # --- color palette: tab20 + grey for noise (-1), for consistency ---
    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgb(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgb({r},{g},{b})"

    fig = go.Figure()

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]

        if c == -1:
            color = noise_color
        else:
            color = _to_rgb(base_palette[i % len(base_palette)])

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

    if method_name == "pca_gmm":
        title = "2D Representation of GMM Clusters"
    else:
        title = "2D Representation of KMeans Clusters"

    fig.update_layout(
        title=title,
        xaxis=dict(title="Principal Component 1", domain=[0.0, 0.65]),
        yaxis=dict(title="Principal Component 2"),
        hovermode="closest",
        # width=950,
        # height=600,
        autosize=True,
        margin=dict(l=60, r=230, t=60, b=60),
        legend=dict(
            title=dict(text="Clusters"),
            x=0.66,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    # === JS: fixed top-right stats panel (works inside Streamlit iframe) ===
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

        // Fixed relative to iframe viewport
        panel.style.position = 'fixed';
        panel.style.top = '80px';
        panel.style.right = '30px';
        panel.style.width = '230px';
        panel.style.maxHeight = '380px';
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
        btn.textContent = '\\u2212';  // minus sign
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

    html_str = fig.to_html(full_html=True, include_plotlyjs="inline")
    insert_pos = html_str.rfind("</body>")
    if (insert_pos != -1):
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive PCA cluster explorer saved → {save_path}")


# ===================== T-SNE / HDBSCAN VISUALS =====================

def plot_tsne_projection(X_tsne, df, color_col, save_path, title=None):
    """Scatterplot of t-SNE projection colored by cluster or target, with distinct palettes."""
    plt.figure(figsize=(8, 6))

    # === Choose palette and title dynamically ===
    if color_col == "td_or_asd":
        palette, base_title = _get_td_asd_palette_and_title(df, "td_or_asd")
        plot_title = base_title.replace("2D", "2D")  # keep wording
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        palette = sns.color_palette("tab20", n_clusters)
        plot_title = "2D Representation of HDBSCAN Clusters"
    else:
        palette = "husl"
        plot_title = title or f"2D Representation of {color_col}"

    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df[color_col],
        palette=palette, s=70, alpha=0.9,
        edgecolor="black", linewidth=0.4
    )

    plt.title(plot_title, fontsize=13, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(
        title=color_col,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True, facecolor='white', framealpha=0.6
    )
    plt.tight_layout(pad=1.2)
    save_dual_format(save_path)


def plot_hdbscan_probability(X_tsne, clusterer, save_path):
    """
    Visualizes HDBSCAN cluster membership strength (probability per point).
    Darker = more confident cluster assignment.
    """
    if not hasattr(clusterer, "probabilities_"):
        print("⚠️ HDBSCAN clusterer has no probabilities_ attribute.")
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=clusterer.probabilities_,
        cmap="viridis",
        s=50, edgecolor='k', alpha=0.8
    )
    plt.colorbar(scatter, label="Cluster Membership Probability")
    plt.title("HDBSCAN Cluster Membership Confidence", fontsize=13, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_tsne_projection_streamlit(X_tsne, df, color_col, save_path, title=None):
    """
    Interactive t-SNE projection for Streamlit.
    - If color_col == 'td_or_asd': use light green/red and TD/ASD legend labels.
    """
    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]

    color_map = None
    legend_title = color_col
    color_field = color_col

    if color_col == "td_or_asd":
        label_map = {0: "TD", 1: "ASD"}
        df_plot["td_asd_label"] = df_plot["td_or_asd"].map(label_map).fillna("Unknown")
        color_field = "td_asd_label"
        legend_title = "Group"
        color_map = {
            "TD": "#9FE2BF",   # soft green
            "ASD": "#FF9999",  # soft red
            "Unknown": "#AAAAAA",
        }

        uniq = set(df_plot["td_or_asd"].dropna().unique())
        if uniq == {0, 1}:
            plot_title = "2D Representation of TD/ASD Participants"
        elif uniq == {1}:
            plot_title = "2D Representation of ASD Participants"
        elif uniq == {0}:
            plot_title = "2D Representation of TD Participants"
        else:
            plot_title = "2D Representation of Participants"

    elif color_col == "cluster":
        color_map = None
        legend_title = "Cluster"
        plot_title = "2D Representation of HDBSCAN Clusters"
    else:
        color_map = None
        plot_title = title or f"2D Representation of {color_col}"

    fig = px.scatter(
        df_plot,
        x="tSNE1",
        y="tSNE2",
        color=color_field,
        color_discrete_map=color_map,
        hover_data={
            "cluster": "cluster" in df_plot.columns,
            "td_or_asd": "td_or_asd" in df_plot.columns,
            "tSNE1": ':.3f',
            "tSNE2": ':.3f',
        },
        title=plot_title,
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(
            size=7,
            line=dict(width=0.8, color="black")  # black border
        )
    )
    fig.update_layout(
        height=600,
        title_font=dict(size=16),
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title=legend_title,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ Streamlit-ready interactive t-SNE saved: {html_path}")


def plot_tsne_cluster_explorer(X_tsne, df, numeric_cols, save_path):
    """
    Interactive t-SNE cluster explorer (HDBSCAN):
    - Scatter by t-SNE1 / t-SNE2
    - Fixed top-right stats panel (Streamlit-friendly)
    """
    save_path = Path(save_path)

    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]

    clusters = sorted(df_plot["cluster"].unique())

    # --- color palette: tab20 + grey for noise (-1) ---
    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgb(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgb({r},{g},{b})"

    fig = go.Figure()

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]

        if c == -1:
            color = noise_color
        else:
            color = _to_rgb(base_palette[i % len(base_palette)])

        fig.add_trace(
            go.Scatter(
                x=sub["tSNE1"],
                y=sub["tSNE2"],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(
                    size=7,
                    color=color,
                    line=dict(width=0.8, color="black"),  # black border around each dot
                ),
                hovertemplate=(
                    "tSNE1: %{x:.3f}<br>"
                    "tSNE2: %{y:.3f}<br>"
                    f"Cluster: {c}<extra></extra>"
                ),
            )
        )

    # --- stats text for each cluster ---
    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_cluster = clusters[0]
    first_key = str(first_cluster)

    # fig.update_layout(
    #     title="2D Representation of HDBSCAN Clusters",
    #     xaxis=dict(title="t-SNE Dimension 1", domain=[0.0, 0.65]),
    #     yaxis=dict(title="t-SNE Dimension 2"),
    #     hovermode="closest",
    #     width=950,
    #     height=600,
    #     margin=dict(l=60, r=230, t=60, b=60),
    #     legend=dict(
    #         title=dict(text="Clusters"),
    #         x=0.66,   # inside middle column [0.60–0.75]
    #         y=0.98,
    #         xanchor="left",
    #         yanchor="top",
    #         bgcolor="rgba(255,255,255,0.85)",
    #         bordercolor="lightgray",
    #         borderwidth=1,
    #         font=dict(size=11),
    #     ),
    # )

    fig.update_layout(
        title="2D Representation of HDBSCAN Clusters",
        xaxis=dict(title="t-SNE Dimension 1"),
        yaxis=dict(title="t-SNE Dimension 2"),
        hovermode="closest",
        # width=950,
        # height=600,
        autosize=True,
        margin=dict(l=60, r=220, t=60, b=60),
        legend=dict(
            title=dict(text="Clusters"),
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    # === JS: fixed top-right stats panel ===
    js_stats = json.dumps(stats_text_map)

    post_script = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel';

        panel.style.position = 'fixed';
        panel.style.top = '80px';
        panel.style.right = '30px';
        panel.style.width = '230px';
        panel.style.maxHeight = '380px';
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
        btn.textContent = '\\u2212';  // minus sign
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

        // ---- update stats when a legend item is clicked ----
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

    html_str = fig.to_html(full_html=True, include_plotlyjs="inline")
    insert_pos = html_str.rfind("</body>")
    if (insert_pos != -1):
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive t-SNE cluster explorer saved → {save_path}")


def plot_pca_cluster_explorer_3d(X_pca, df, numeric_cols, save_path, method_name="kmeans"):
    """
    Interactive PCA cluster explorer in 3D:
    - Scatter by PC1 / PC2 / PC3
    - Each cluster is a separate trace
    - Fixed top-right stats panel (same behavior as 2D explorer)

    Only works if X_pca has at least 3 components.
    """
    save_path = Path(save_path)

    if X_pca.shape[1] < 3:
        print("⚠️ plot_pca_cluster_explorer_3d: X_pca has < 3 components, skipping 3D explorer.")
        return

    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]
    df_plot["PC3"] = X_pca[:, 2]

    clusters = sorted(df_plot["cluster"].unique())

    # --- color palette: tab20 + grey for noise (-1) ---
    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgb(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgb({r},{g},{b})"

    fig = go.Figure()

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]

        if c == -1:
            color = noise_color
        else:
            color = _to_rgb(base_palette[i % len(base_palette)])

        fig.add_trace(
            go.Scatter3d(
                x=sub["PC1"],
                y=sub["PC2"],
                z=sub["PC3"],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(
                    size=5,
                    color=color,
                    line=dict(width=0.7, color="black"),
                ),
                hovertemplate=(
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "PC3: %{z:.3f}<br>"
                    f"Cluster: {c}<extra></extra>"
                ),
            )
        )

    # --- stats text for each cluster ---
    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_cluster = clusters[0]
    first_key = str(first_cluster)

    if method_name == "pca_gmm":
        title = "3D Representation of GMM Clusters"
    else:
        title = "3D Representation of KMeans Clusters"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3",
        ),
        hovermode="closest",
        width=950,
        height=650,
        margin=dict(l=60, r=230, t=60, b=60),
        legend=dict(
            title=dict(text="Clusters"),
            x=0.66,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    # === JS: fixed top-right stats panel, same as 2D ===
    js_stats = json.dumps(stats_text_map)

    post_script = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel-3d';

        panel.style.position = 'fixed';
        panel.style.top = '80px';
        panel.style.right = '30px';
        panel.style.width = '230px';
        panel.style.maxHeight = '380px';
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
        title.textContent = 'Cluster stats (3D)';

        var btn = document.createElement('button');
        btn.textContent = '\\u2212';  // minus sign
        btn.style.marginLeft = '8px';
        btn.style.cursor = 'pointer';
        btn.style.border = '1px solid #888';
        btn.style.background = '#f7f7f7';
        btn.style.padding = '0 6px';

        header.appendChild(title);
        header.appendChild(btn);

        var content = document.createElement('div');
        content.id = 'cluster-stats-content-3d';
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

    html_str = fig.to_html(full_html=True, include_plotlyjs="inline")
    insert_pos = html_str.rfind("</body>")
    if insert_pos != -1:
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive 3D PCA cluster explorer saved → {save_path}")


def plot_tsne_cluster_explorer_3d(X_tsne, df, numeric_cols, save_path):
    """
    Interactive t-SNE cluster explorer in 3D (HDBSCAN):
    - Scatter by tSNE1 / tSNE2 / tSNE3
    - Fixed top-right stats panel (same behavior as 2D explorer)

    Only works if X_tsne has at least 3 dimensions.
    """
    save_path = Path(save_path)

    if X_tsne.shape[1] < 3:
        print("⚠️ plot_tsne_cluster_explorer_3d: X_tsne has < 3 dims, skipping 3D explorer.")
        return

    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]
    df_plot["tSNE3"] = X_tsne[:, 2]

    clusters = sorted(df_plot["cluster"].unique())

    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgb(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgb({r},{g},{b})"

    fig = go.Figure()

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]

        if c == -1:
            color = noise_color
        else:
            color = _to_rgb(base_palette[i % len(base_palette)])

        fig.add_trace(
            go.Scatter3d(
                x=sub["tSNE1"],
                y=sub["tSNE2"],
                z=sub["tSNE3"],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(
                    size=5,
                    color=color,
                    line=dict(width=0.7, color="black"),
                ),
                hovertemplate=(
                    "tSNE1: %{x:.3f}<br>"
                    "tSNE2: %{y:.3f}<br>"
                    "tSNE3: %{z:.3f}<br>"
                    f"Cluster: {c}<extra></extra>"
                ),
            )
        )

    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_cluster = clusters[0]
    first_key = str(first_cluster)

    fig.update_layout(
        title="3D Representation of HDBSCAN Clusters",
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
        ),
        hovermode="closest",
        width=950,
        height=650,
        margin=dict(l=60, r=230, t=60, b=60),
        legend=dict(
            title=dict(text="Clusters"),
            x=0.66,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    js_stats = json.dumps(stats_text_map)

    post_script = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel-3d-tsne';

        panel.style.position = 'fixed';
        panel.style.top = '80px';
        panel.style.right = '30px';
        panel.style.width = '230px';
        panel.style.maxHeight = '380px';
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
        title.textContent = 'Cluster stats (3D)';

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
        content.id = 'cluster-stats-content-3d-tsne';
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

    html_str = fig.to_html(full_html=True, include_plotlyjs="inline")
    insert_pos = html_str.rfind("</body>")
    if insert_pos != -1:
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive 3D t-SNE cluster explorer saved → {save_path}")


# ===================== NEW: T-SNE 3D PROJECTIONS (STATIC + STREAMLIT) =====================

def plot_tsne_projection_3d(X_tsne, df, color_col, save_path, title=None):
    """
    Static 3D t-SNE projection (tSNE1, tSNE2, tSNE3)
    for either clusters or TD/ASD target.
    """
    if X_tsne.shape[1] < 3:
        print("⚠️ plot_tsne_projection_3d: X_tsne has < 3 dims, skipping.")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if color_col == "td_or_asd":
        palette, base_title = _get_td_asd_palette_and_title(df, "td_or_asd")
        colors = df[color_col].map(palette)
        plot_title = base_title.replace("2D", "3D")
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        base_palette = sns.color_palette("tab20", n_clusters)
        color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(sorted(df[color_col].unique()))}
        colors = df[color_col].map(color_map)
        plot_title = "3D Representation of HDBSCAN Clusters"
    else:
        pal = sns.color_palette("husl", df[color_col].nunique())
        color_map = {c: pal[i % len(pal)] for i, c in enumerate(sorted(df[color_col].unique()))}
        colors = df[color_col].map(color_map)
        plot_title = f"3D Representation of {color_col}"

    final_title = title or plot_title

    ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        X_tsne[:, 2],
        c=colors,
        s=45,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.set_title(final_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")

    plt.tight_layout()
    save_dual_format(save_path)


def plot_tsne_projection_streamlit_3d(X_tsne, df, color_col, save_path, title=None):
    """
    Interactive 3D t-SNE projection for Streamlit (tSNE1, tSNE2, tSNE3).
    Supports both cluster color and TD/ASD target color.
    """
    if X_tsne.shape[1] < 3:
        print("⚠️ plot_tsne_projection_streamlit_3d: X_tsne has < 3 dims, skipping.")
        return

    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]
    df_plot["tSNE3"] = X_tsne[:, 2]

    color_map = None
    legend_title = color_col
    color_field = color_col

    if color_col == "td_or_asd":
        label_map = {0: "TD", 1: "ASD"}
        df_plot["td_asd_label"] = df_plot["td_or_asd"].map(label_map).fillna("Unknown")
        color_field = "td_asd_label"
        legend_title = "Group"
        color_map = {
            "TD": "#9FE2BF",
            "ASD": "#FF9999",
            "Unknown": "#AAAAAA",
        }

        _, base_title = _get_td_asd_palette_and_title(df_plot, "td_or_asd")
        plot_title = base_title.replace("2D", "3D")
    elif color_col == "cluster":
        legend_title = "Cluster"
        plot_title = "3D Representation of HDBSCAN Clusters"
    else:
        plot_title = title or f"3D Representation of {color_col}"

    fig = px.scatter_3d(
        df_plot,
        x="tSNE1",
        y="tSNE2",
        z="tSNE3",
        color=color_field,
        color_discrete_map=color_map,
        hover_data={
            "cluster": "cluster" in df_plot.columns,
            "td_or_asd": "td_or_asd" in df_plot.columns,
            "tSNE1": ':.3f',
            "tSNE2": ':.3f',
            "tSNE3": ':.3f',
        },
        title=plot_title,
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(
            size=5,
            line=dict(width=0.7, color="black")
        )
    )
    fig.update_layout(
        height=650,
        title_font=dict(size=16),
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
        ),
        legend_title=legend_title,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ Streamlit-ready interactive 3D t-SNE saved: {html_path}")


# ===================== NEW: JSD-BASED VISUALS FOR TABS =====================

def plot_jsd_cluster_heatmap(jsd_stats, save_path, title="JSD-based Cluster Separation"):
    """
    JSD heatmap across clusters (aggregated over features).

    jsd_stats: dict from metrics["jsd_stats"]:
      {
        "cluster_labels": [...],
        "per_feature": {feat: [[...], [...], ...]},
        ...
      }

    We compute:
      J_ij = mean over features of JSD(feat, cluster_i vs cluster_j)
    and plot J as a k x k heatmap.
    """
    save_path = Path(save_path)

    cluster_labels = jsd_stats.get("cluster_labels", [])
    per_feature = jsd_stats.get("per_feature", {})

    k = len(cluster_labels)
    if k == 0 or not per_feature:
        print("⚠️ plot_jsd_cluster_heatmap: missing clusters or per_feature; skipping.")
        return

    mat_sum = np.zeros((k, k), dtype=float)
    n_feats = 0

    for feat, m in per_feature.items():
        arr = np.asarray(m, dtype=float)
        if arr.shape != (k, k):
            continue
        mat_sum += arr
        n_feats += 1

    if n_feats == 0:
        print("⚠️ plot_jsd_cluster_heatmap: no valid feature matrices; skipping.")
        return

    mat_mean = mat_sum / n_feats
    df_heat = pd.DataFrame(
        mat_mean,
        index=[str(c) for c in cluster_labels],
        columns=[str(c) for c in cluster_labels],
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".3f",
        cmap="magma",
        square=True,
        cbar_kws={"label": "Mean JSD across features"},
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    save_dual_format(save_path)
    print(f"✅ JSD cluster heatmap saved → {save_path}")


def plot_jsd_feature_ranking(jsd_stats, save_path, top_n=15, title="Top Features by JSD Separation"):
    """
    Bar chart of top_n features by mean JSD score.

    jsd_stats: dict from metrics["jsd_stats"]:
      {
        "feature_scores": {feat: float},
        "ranked_features": [...]
      }
    """
    save_path = Path(save_path)

    feature_scores = jsd_stats.get("feature_scores", {})
    if not feature_scores:
        print("⚠️ plot_jsd_feature_ranking: feature_scores missing/empty; skipping.")
        return

    # Use ranked_features if given, else sort from feature_scores
    ranked = jsd_stats.get("ranked_features")
    if ranked:
        ranked = [f for f in ranked if f in feature_scores]
    else:
        ranked = sorted(feature_scores.keys(), key=lambda f: feature_scores[f], reverse=True)

    ranked = ranked[:top_n]
    scores = [feature_scores[f] for f in ranked]

    plt.figure(figsize=(8, max(4, 0.3 * len(ranked) + 2)))
    y_pos = np.arange(len(ranked))

    plt.barh(y_pos, scores, color="teal")
    plt.yticks(y_pos, ranked)
    plt.gca().invert_yaxis()  # highest JSD at top
    plt.xlabel("Mean JSD across cluster pairs")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_dual_format(save_path)
    print(f"✅ JSD feature ranking plot saved → {save_path}")
