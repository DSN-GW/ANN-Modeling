import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
import json



# ===================== HELPER FUNCTION =====================

def save_dual_format(save_path):
    """Save both PNG and PDF versions of the plot for paper-quality exports."""
    plt.savefig(save_path, dpi=300)
    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300)
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


def plot_tsne_projection(X_tsne, df, color_col, save_path, title=None):
    """Scatterplot of t-SNE projection colored by cluster or target, with distinct palettes."""
    plt.figure(figsize=(8, 6))

    # === Choose palette and title dynamically ===
    if color_col == "td_or_asd":
        palette = {0: "#9FE2BF", 1: "#FF9999"}  # TD = green, ASD = red (soft)
        if df[color_col].nunique() == 1:
            plot_title = "2D Representation of ASD Participants"
        else:
            plot_title = "2D Representation of TD/ASD Participants"
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


# ===================== HDBSCAN-SPECIFIC VISUALS =====================

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


# ===================== FEATURE IMPORTANCE VISUALS =====================

def plot_cluster_feature_means(df, features, save_path, method_name="tsne_hdbscan"):
    """Visualizes mean feature values per cluster (centroids)."""
    cluster_means = df[df["cluster"] != -1].groupby("cluster")[features].mean()
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm")
    plt.title(f"Cluster Centroids by Feature ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_dual_format(save_path)


def plot_feature_variance(df, features, save_path, method_name="tsne_hdbscan"):
    """Plots variance of feature means across clusters."""
    feature_variance = df[df["cluster"] != -1].groupby("cluster")[features].mean().var()
    plt.figure(figsize=(8, 4))
    feature_variance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title(f"Feature Variance Across Clusters ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Variance of Cluster Means")
    plt.tight_layout()
    save_dual_format(save_path)


# ===================== STREAMLIT VISUALS =====================

def plot_tsne_projection_streamlit(X_tsne, df, color_col, save_path, title=None):
    """
    Generates interactive t-SNE projection for Streamlit and saves as .html.
    Hover shows only: cluster, td_or_asd, t-SNE dims.
    """
    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]

    # === Color and title logic ===
    if color_col == "td_or_asd":
        color_map = {0: "#9FE2BF", 1: "#FF9999"}  # TD = green, ASD = red
        if df_plot[color_col].nunique() == 1:
            plot_title = "2D Representation of ASD Participants"
        else:
            plot_title = "2D Representation of TD/ASD Participants"
    elif color_col == "cluster":
        color_map = None
        plot_title = "2D Representation of HDBSCAN Clusters"
    else:
        color_map = None
        plot_title = title or f"2D Representation of {color_col}"

    fig = px.scatter(
        df_plot,
        x="tSNE1", y="tSNE2",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={
            "cluster": True if "cluster" in df_plot.columns else False,
            "td_or_asd": True if "td_or_asd" in df_plot.columns else False,
            "tSNE1": ':.3f',
            "tSNE2": ':.3f'
        },
        title=plot_title,
        template="plotly_white"
    )

    fig.update_layout(
        height=600,
        title_font=dict(size=16),
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title=color_col,
    )

    # === Save as .html for Streamlit ===
    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs='inline')
    print(f"✅ Streamlit-ready interactive t-SNE saved: {html_path}")

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
    lines.append("")  # blank line

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
        lines.append("")  # blank line between features

    # join with <br> for HTML
    return "<br>".join(lines)

# def plot_tsne_cluster_explorer(X_tsne, df, numeric_cols, save_path):
#     """
#     Interactive t-SNE cluster explorer for V3:
#     - Scatter by t-SNE1 / t-SNE2
#     - Each cluster is a separate trace
#     - Right-side annotation shows descriptive stats for the selected cluster
#     - Clicking a cluster in the legend updates the stats panel
#     """
#     save_path = Path(save_path)
#
#     df_plot = df.copy()
#     df_plot["tSNE1"] = X_tsne[:, 0]
#     df_plot["tSNE2"] = X_tsne[:, 1]
#
#     clusters = sorted(df_plot["cluster"].unique())
#
#     # --- color palette: tab20 + grey for noise (-1) ---
#     base_palette = sns.color_palette("tab20", 20)
#     noise_color = "#999999"
#
#     def _to_rgba(col):
#         r, g, b = [int(255 * c) for c in col]
#         return f"rgba({r},{g},{b},1)"
#
#     fig = go.Figure()
#
#     for i, c in enumerate(clusters):
#         sub = df_plot[df_plot["cluster"] == c]
#
#         if c == -1:
#             color = noise_color
#         else:
#             color = _to_rgba(base_palette[i % len(base_palette)])
#
#         fig.add_trace(
#             go.Scatter(
#                 x=sub["tSNE1"],
#                 y=sub["tSNE2"],
#                 mode="markers",
#                 name=f"Cluster {c}",
#                 marker=dict(
#                     size=7,
#                     color=color,
#                     line=dict(width=0.8, color="black")  # ← border around each dot
#                 ),
#                 hovertemplate=(
#                     "tSNE1: %{x:.3f}<br>"
#                     "tSNE2: %{y:.3f}<br>"
#                     f"Cluster: {c}<extra></extra>"
#                 ),
#             )
#         )
#
#     # --- stats text for each cluster ---
#     stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
#     first_cluster = clusters[0]
#     initial_text = stats_text_map[str(first_cluster)]
#
#     fig.update_layout(
#         title="2D Representation of HDBSCAN Clusters",
#
#         # 60% scatterplot on left
#         xaxis=dict(title="t-SNE Dimension 1", domain=[0.0, 0.60]),
#
#         yaxis=dict(title="t-SNE Dimension 2"),
#         hovermode="closest",
#         width=1200,
#         height=700,
#
#         margin=dict(l=60, r=120, t=60, b=60),
#
#         # Legend in the middle column (15%)
#         legend=dict(
#             title="Clusters",
#             x=0.62,  # inside middle column [0.60–0.75]
#             y=0.98,
#             xanchor="left",
#             yanchor="top",
#             bgcolor="rgba(255,255,255,0.85)",
#             bordercolor="lightgray",
#             borderwidth=1,
#         )
#     )
#
#     # Stats panel in the right-most 25%
#     fig.add_annotation(
#         text=initial_text,
#         align="left",
#         showarrow=False,
#         xref="paper",
#         yref="paper",
#         x=0.80,  # right column = stats panel
#         y=0.98,
#         xanchor="left",
#         yanchor="top",
#         bordercolor="black",
#         borderwidth=1,
#         bgcolor="white",
#         opacity=0.95,
#         font=dict(size=12),
#     )
#
#     # --- JS hook to update stats on legend click ---
#     js_stats = json.dumps(stats_text_map)
#
#     post_script = f"""
#     <script>
#     document.addEventListener("DOMContentLoaded", function() {{
#         var statsByCluster = {js_stats};
#         var plot = document.querySelectorAll('div.js-plotly-plot')[0];
#
#         plot.on('plotly_legendclick', function(event) {{
#             var trace = event.data[event.curveNumber];
#             var name = trace.name || "";
#             var cid = name.replace("Cluster ", "");
#             var text = statsByCluster[cid];
#
#             if (text) {{
#                 Plotly.relayout(plot, {{
#                     'annotations[0].text': text
#                 }});
#             }}
#         }});
#     }});
#     </script>
#     """
#
#     # ---- generate HTML first, then inject our JS before </body> ----
#     html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
#
#     insert_pos = html_str.rfind("</body>")
#     if insert_pos != -1:
#         html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
#     else:
#         html_str += post_script
#
#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write(html_str)
#
#     print(f"✅ Interactive t-SNE cluster explorer saved → {save_path}")


def plot_tsne_cluster_explorer(X_tsne, df, numeric_cols, save_path):
    """
    Interactive t-SNE cluster explorer for V3:
    - Scatter by t-SNE1 / t-SNE2
    - Each cluster is a separate trace
    - Right-side panel shows descriptive stats for the selected cluster
    - Clicking a cluster in the legend updates the stats panel
    """
    save_path = Path(save_path)

    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]

    clusters = sorted(df_plot["cluster"].unique())

    # --- color palette: tab20 + grey for noise (-1) ---
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

    # ---- layout: 60% scatter, 15% legend column, 25% free for stats panel ----
    fig.update_layout(
        title="2D Representaion of HDBSCAN Clusters",
        xaxis=dict(title="t-SNE Dimension 1", domain=[0.0, 0.60]),
        yaxis=dict(title="t-SNE Dimension 2"),
        hovermode="closest",
        width=1200,
        height=700,
        margin=dict(l=60, r=200, t=60, b=60),
        legend=dict(
            title="Clusters",
            x=0.62,   # inside middle column [0.60–0.75]
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
    )

    # === JS: external scrollable + collapsible stats panel ===
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

        // Position to the right of the figure (roughly last 25%)
        var rect = plot.getBoundingClientRect();
        panel.style.position = 'absolute';
        panel.style.top = (rect.top + window.scrollY + 40) + 'px';
        panel.style.left = (rect.left + window.scrollX + rect.width * 0.75) + 'px';
        panel.style.width = '260px';
        panel.style.maxHeight = '500px';
        panel.style.overflowY = 'auto';      // scrollable
        panel.style.border = '1px solid #333';
        panel.style.background = 'white';
        panel.style.padding = '8px 10px';
        panel.style.boxShadow = '0 0 4px rgba(0,0,0,0.2)';
        panel.style.fontFamily = 'Arial, sans-serif';
        panel.style.fontSize = '12px';
        panel.style.boxSizing = 'border-box';
        panel.style.zIndex = 1000;

        // ---- header with collapse/expand button ----
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

    # ---- generate HTML and inject JS before </body> ----
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    insert_pos = html_str.rfind("</body>")
    if insert_pos != -1:
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive t-SNE cluster explorer saved → {save_path}")
