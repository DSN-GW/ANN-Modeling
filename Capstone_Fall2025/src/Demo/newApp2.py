import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
from importlib import import_module
import plotly.figure_factory as ff
import streamlit.components.v1 as components

# ----- App / Layout -----
st.set_page_config(
    page_title="TD/ASD Classification - Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Paths -----
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
DATA_ROOT = PROJECT_ROOT / "data"

# Ensure Root/src is on sys.path so Model_V1, Model_V2, etc. can be imported
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ----- Imports for clustering page -----
from Clustering.general_clustering_model import (
    GeneralClusterAnalyzer,
    NLP_FEATURES,
    DATA_VERSION as CLUSTER_DATA_VERSION,
)
from Clustering.data_preprocessor import preprocess_clustering_data


MODEL_OVERVIEW_MD = {
    "V1": """
## Model Overview (V1)

Baseline **XGBoost** classifier on core NLP features.

### Key Features
- Token/char stats, sentiment, readability
- Basic linguistic cohesion features
- SHAP + LIME for explainability

### Pipeline
1. Preprocess text
2. Feature engineering (core NLP)
3. XGBoost classification
4. Explainability (global + local)
5. Visualization
""",

    "V2": """
## Model Overview (V2)

Adds **characteristic-based** features (11 traits) extracted via LLM.

### Key Features
- All V1 features
- + 11 characteristic scores from LLM
- Improved SHAP global consistency

### Pipeline
1. Preprocess + LLM traits
2. Feature engineering (core + traits)
3. XGBoost classification
4. Explainability
5. Visualization
""",

    "V3": """
## Model Overview (V3)

Refined traits + stronger regularization; calibrated probabilities.

### Key Features
- All V2 features
- Probability calibration (Platt/Isotonic)
- Better class balance handling

### Pipeline
1. Preprocess + refined traits
2. Feature engineering
3. XGBoost + calibration
4. Explainability
5. Visualization
""",

    "V4": """
## Model Overview (V4)

- Data Source : Data_V4
- Train-Test Split : 80-20 Stratified Split with Seed 42
- Key Change : Generated Slope Features

""",

    "V7-1": """
## Model Overview (V7.1)

- Data Source : Data_V7
- Features Added : TDNorm PE and Concept Learning
- Train-Test Split : 80-20 Stratified Split with Seed 42
- Baseline Model for FSR overlap Test
""",

    "V7-2": """
## Model Overview (V7.2)

- Data Source : Data_V7
- Features Added : TDNorm PE and Concept Learning
- Train-Test Split : FSR Overlap Region is Training Set
- Baseline Model for FSR overlap Test
""",

    "V7-3": """
## Model Overview (V7.3)

- Data Source : Data_V7
- Features Added : TDNorm PE and Concept Learning
- Train-Test Split : FSR Non-Overlap Region is Training Set
- Baseline Model for FSR overlap Test
"""
}

MODEL_INFO = {
    "V1": "Baseline TD/ASD classifier using core NLP and linguistic features.",
    "V2": "Enhanced model with characteristic-based features extracted via LLM.",
    "V3": "Advanced TD/ASD Classification using XGBoost and NLP features.",
    "V4": "Cross-validated model with feature stability and improved explainability.",
    "V5": "Production-ready model with calibrated thresholds and drift detection."
}


def get_model_info(version: str) -> str:
    """Return sidebar caption text for a given model version."""
    return MODEL_INFO.get(version, "Experimental TD/ASD model.")


def _discover_models() -> list[str]:
    """Return available model versions by scanning Results/* that look like V{number}."""
    if not RESULTS_ROOT.exists():
        return ["V1", "V2", "V3", "V4", "V5"]
    versions = []
    for p in sorted(RESULTS_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("V") and p.name[1].isdigit():
            versions.append(p.name.upper())
    return versions or ["V1", "V2", "V3", "V4", "V5"]


def _discover_culstering_models() -> list[str]:
    """Return available clustering model versions by scanning Results/* that look like Clustering_V{number}."""
    if not RESULTS_ROOT.exists():
        return ["Clustering_V1", "Clustering_V2"]
    versions = []
    for p in sorted(RESULTS_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("CLUSTERING"):
            versions.append(p.name.upper())
    return versions or ["Clustering_V1", "Clustering_V2"]


def _discover_datasets() -> list[str]:
    """Return available datasets by scanning data/* that look like Data_v{number}."""
    if not DATA_ROOT.exists():
        return ["V1", "V2", "V3"]
    versions = []
    for p in sorted(DATA_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("DATA_V"):
            versions.append(p.name.upper())
    return versions or ["V1", "V2", "V3", "V4"]


def get_overview_md(version: str) -> str:
    """Return Model MD Card based on Model Version Selection."""
    return MODEL_OVERVIEW_MD.get(version, MODEL_OVERVIEW_MD["V1"])


# Initialize session state and model selection
if "page" not in st.session_state:
    st.session_state.page = "home"

if "model_version" not in st.session_state:
    st.session_state.model_version = "V7.1"

if "data_version" not in st.session_state:
    st.session_state.data_version = "DATA_V7.1"

if "available_models" not in st.session_state:
    st.session_state.available_models = _discover_models()

if "available_datasets" not in st.session_state:
    st.session_state.available_datasets = _discover_datasets()

# For PCA preview stage
if "pca_preview_dir" not in st.session_state:
    st.session_state.pca_preview_dir = ""
if "pca_preview_subset_mode" not in st.session_state:
    st.session_state.pca_preview_subset_mode = None
if "pca_preview_feature_mode" not in st.session_state:
    st.session_state.pca_preview_feature_mode = None
if "pca_preview_model_type" not in st.session_state:
    st.session_state.pca_preview_model_type = None


class DemoApp:
    def __init__(self):
        self.model_version = st.session_state.model_version
        self.data_version = st.session_state.data_version
        self.results_dir = RESULTS_ROOT / self.model_version
        self.data_dir = DATA_ROOT / self.model_version
        self.viz_dir = self.results_dir / "visualizations"
        self.predictor = None

    # ----- Model Version Selection -----
    def set_model_version(self, version: str):
        """Update the app to a new model version and refresh result paths."""
        self.model_version = version
        st.session_state.model_version = version
        self.results_dir = RESULTS_ROOT / version
        self.data_dir = DATA_ROOT / self.model_version
        self.viz_dir = self.results_dir / "visualizations"

    def set_data_version(self, version: str):
        """Update the app to a new data version and refresh paths."""
        self.data_version = version
        st.session_state.data_version = version
        self.data_dir = DATA_ROOT / self.data_version

    # ----- Data / Model loading  -----
    def load_results(self) -> bool:
        try:
            version = self.model_version
            explainability_path = self.results_dir / f'explainability_analysis_{version}.json'
            training_results_path = self.results_dir / f'training_results_{version}.json'
            test_results_path = self.results_dir / f'test_results_{version}.json'

            with open(explainability_path, 'r') as f:
                self.explainability_data = json.load(f)

            with open(training_results_path, 'r') as f:
                self.training_results = json.load(f)

            try:
                with open(test_results_path, 'r') as f:
                    self.test_results = json.load(f)
            except Exception:
                self.test_results = None

            return True

        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            st.error(f"Results directory: {self.results_dir}")
            st.error(f"Directory exists: {self.results_dir.exists()}")
            if self.results_dir.exists():
                st.error(f"Files in directory: {list(self.results_dir.glob('*.json'))}")
            return False

    def initialize_predictor(self):
        # Load model + artifacts for the selected version here
        return

    # ----- Pages -----
    def show_home_page(self):
        st.header("🏠 About the Project")

        st.markdown("""
            ### Project Overview
            This project aims to investigate how **interpretable,
            multi-modal machine learning models** can enhance the
            diagnosis and understanding of **Autism Spectrum Disorder
            (ASD)**. By integrating **behavioral** and **linguistic
            data**, the study aims to move beyond traditional, subjective
            assessments toward more objective, data-driven approaches.
            The proposed models seek to improve **diagnosis precision**
            by identifying subtle patterns across multiple data types, enable
            **subtype discovery** through clustering of behavioral and linguistic
            traits, and enhance **generalizability** by developing frameworks that
            perform robustly across diverse populations and datasets. Ultimately,
            this research contributes to more accurate, transparent, and inclusive
            ASD prediction and analysis.

            ---
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Key Objectives")
            st.markdown("""
            - 🧩 Develop interpretable machine learning models  
            - 🗣️ Analyze behavioral and linguistic patterns  
            - 🧠 Identify subtypes of ASD using clustering  
            - 📈 Improve diagnostic generalization across datasets
            """)

        with col2:
            st.subheader("🧰 Tools & Techniques")
            st.markdown("""
            - **LLM Agent:** Qwen  
            - **NLP:** BERT  
            - **Classification Model:** XGBoost  
            - **Dimension Reduction:** PCA, AGAE  
            - **Clustering Model:** KMeans, Gaussian Mixture Model, HDBSCAN  
            - **Explainability Method:** SHAP, LIME
            """)

        st.markdown("---")
        st.subheader("🧭 App Navigation Guide")

        st.markdown("""
        Use the tabs on the left to explore different parts of the project:

        - **🏠 Home** – Project overview, key objectives, tools and techniques  
        - **📈 Data & Model** – Data versions, data dictionary, models selection  
        - **📊 EDA** – Exploratory Data Analysis for the chosen features  
        - **🧮 Model Results** – Train accuracy, test accuracy, confusion matrix, F1-scores  
        - **🧩 Cluster Analysis** – Unsupervised clustering (KMeans, GMM, HDBSCAN) and ASD subtype discovery  
        - **🔍 xAI - Explainability** – SHAP / LIME explanations for model predictions and important features
        """)

    def show_model_and_data_page(self):
        st.header("📈 Data and Model Cards")

        st.markdown("""
            <style>
            /* Center and evenly space Streamlit tabs */
            div[data-baseweb="tab-list"] {
                justify-content: space-evenly;
            }
            </style>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📂 Data Card", "🧮 Model Card"])

        # -------------------------
        # Data card
        # -------------------------
        with tab1:
            st.markdown("## Data Overview")

            st.markdown("""
            The data is provided by The George Washington University - Department of Psychological 
            & Brain Sciences. There are 1,119 participants in this dataset, aged from 8 to 12 years.
            """)

            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            data_card_dir = project_root / "data" / "Data_card"

            dict_v1_path = data_card_dir / "LLM data_trial_dictionary.csv"
            dict_v2_path = data_card_dir / "LLM data dictionary.csv"
            dict_v3_path = data_card_dir / "LLM data_aggregate_dictionary.csv"

            data_v1_path = data_card_dir / "LLM data_trial.csv"
            data_v2_path = data_card_dir / "LLM data.csv"
            data_v3_path = data_card_dir / "LLM data_aggregate.csv"

            # --- Data V1 ---
            with st.expander("Data V1: Raw Trial-Level Data", expanded=False):
                st.write("""
                    - 187,187 observations  
                    - 19 variables  
                    - Contains raw data at the trial level for all participants  
                """)

                st.markdown("#### Data Dictionary – Data V1")
                if not dict_v1_path.exists():
                    st.error("❌ Data V1 dictionary file not found.")
                else:
                    try:
                        dict_v1 = pd.read_csv(dict_v1_path)
                        st.dataframe(dict_v1, use_container_width=True, height=300)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V1 dictionary: {e}")

                st.markdown("#### Data Preview")
                if not data_v1_path.exists():
                    st.error("❌ Data V1 file not found. Check the path or file name.")
                else:
                    try:
                        df_v1 = pd.read_csv(data_v1_path)
                        st.dataframe(df_v1.head(20))
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V1: {e}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Use Data V1 for Modeling", key="use_v1"):
                            self.set_data_version("Data V1")
                            st.success(f"✅ Data version set to **{self.data_version}**.")
                            st.rerun()
                    with col2:
                        if st.button("See EDA for Data V1", key="eda_v1"):
                            st.session_state["page"] = "eda"
                            st.session_state["selected_data_version"] = "Data V1"
                            st.rerun()

            # --- Data V2 ---
            with st.expander("Data V2: Aggregated with Slope Features", expanded=False):
                st.write("""
                    - 2,648 observations  
                    - 11 variables  
                    - Aggregated per participant for each profile with computed slope features (avg_PE) 
                """)

                st.markdown("#### Data Dictionary – Data V2")
                if not dict_v2_path.exists():
                    st.error("❌ Data V2 dictionary file not found.")
                else:
                    try:
                        dict_v2 = pd.read_csv(dict_v2_path)
                        st.dataframe(dict_v2, use_container_width=True, height=300)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V2 dictionary: {e}")

                st.markdown("#### Data Preview")
                if not data_v2_path.exists():
                    st.error("❌ Data V2 file not found.")
                else:
                    try:
                        df_v2 = pd.read_csv(data_v2_path)
                        st.dataframe(df_v2.head(20))
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V2: {e}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use Data V2 for Modeling", key="use_v2"):
                        self.set_data_version("Data V2")
                        st.success(f"✅ Data version set to **{self.data_version}**.")
                        st.rerun()
                with col2:
                    if st.button("See EDA for Data V2", key="eda_v2"):
                        st.session_state["page"] = "eda"
                        st.session_state["selected_data_version"] = "Data V2"
                        st.rerun()

            # --- Data V3 ---
            with st.expander("Data V3: Aggregated with Concept Learning Features", expanded=False):
                st.write("""
                    - 1,119 observations  
                    - 10 variables  
                    - Participant-level dataset with cognitive and behavioral learning features  
                """)

                st.markdown("#### Data Dictionary – Data V3")
                if not dict_v3_path.exists():
                    st.error("❌ Data V3 dictionary file not found.")
                else:
                    try:
                        dict_v3 = pd.read_csv(dict_v3_path)
                        st.dataframe(dict_v3, use_container_width=True, height=300)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V3 dictionary: {e}")

                st.markdown("#### Data Preview")
                if not data_v3_path.exists():
                    st.error("❌ Data V3 file not found.")
                else:
                    try:
                        df_v3 = pd.read_csv(data_v3_path)
                        st.dataframe(df_v3.head(20))
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V3: {e}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use Data V3 for Modeling", key="use_v3"):
                        self.set_data_version("Data V3")
                        st.success(f"✅ Data version set to **{self.data_version}**.")
                        st.rerun()
                with col2:
                    if st.button("See EDA for Data V3", key="eda_v3"):
                        st.session_state["page"] = "eda"
                        st.session_state["selected_data_version"] = "Data V3"
                        st.rerun()

        # -------------------------
        # Model card
        # -------------------------
        with tab2:
            st.markdown("#### Select a Model Version")
            selected = st.selectbox(
                "Choose which model version to explore:",
                options=st.session_state.available_models,
                index=(
                    st.session_state.available_models.index(st.session_state.model_version)
                    if st.session_state.model_version in st.session_state.available_models
                    else 0
                ),
                key="model_version_select",
            )

            if selected != self.model_version:
                self.set_model_version(selected)
                st.success(f"Model version set to **{self.model_version}**.")
                st.rerun()

            st.markdown("### Checklist for Specific Files Needed for Downstream Analysis")

            data = [
                ["Processed Training Data", f"data/Data_{self.model_version}",
                 f"LLM_data_train_preprocessed_{self.model_version}", "", ""],
                ["Processed Test Data", f"data/Data_{self.model_version}",
                 f"LLM_data_test_preprocessed_{self.model_version}", "", ""],
                ["Training Results", f"Results/{self.model_version}",
                 f"training_results_{self.model_version}", "", ""],
                ["Test Results", f"Results/{self.model_version}",
                 f"test_results_{self.model_version}", "", ""],
                ["Model Pkl File", f"Results/{self.model_version}",
                 f"xgboost_model_{self.model_version}", "", ""],
            ]

            df = pd.DataFrame(
                data, columns=["Files", "Directory", "Filename", "Status", "Instruction"]
            )

            for i, row in df.iterrows():
                if i in [0, 1]:
                    ext = ".csv"
                    script_hint = "run data_preprocesor.py"
                elif i in [2, 3]:
                    ext = ".json"
                    script_hint = "run train.py" if i == 2 else "run predict.py"
                else:
                    ext = ".pkl"
                    script_hint = "run train.py"

                path = PROJECT_ROOT / row["Directory"] / (row["Filename"] + ext)
                exists = path.exists()

                df.loc[i, "Status"] = "✅ Found" if exists else "❌ Missing"
                df.loc[i, "Instruction"] = "Proceed" if exists else script_hint

            st.dataframe(df, use_container_width=True)

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_overview_md(self.model_version))

                if st.button("🚀 Explore Model Results"):
                    st.session_state.page = "results"
                    st.rerun()

            with col2:
                st.subheader("Selected Feature List")
                if hasattr(self, "training_results") and self.training_results:
                    features_selected = self.training_results["feature_names"]
                    st.info(f"Number of Features : {self.training_results['n_features']}")
                    st.json(features_selected)
                else:
                    st.info("Training report not found. Run train.py to generate a training report.")

    def show_eda_page(self):
        st.header("📊 Explanatory Data Analysis")

        st.markdown("""
            <style>
            /* Center and evenly space Streamlit tabs */
            div[data-baseweb="tab-list"] {
                justify-content: space-evenly;
            }
            </style>
        """, unsafe_allow_html=True)

        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        data_dir = project_root / "data" / "Data_card"

        version_files = {
            "Data V1": data_dir / "LLM data_trial.csv",
            "Data V2": data_dir / "LLM data.csv",
            "Data V3": data_dir / "LLM data_aggregate.csv",
        }

        selected_version = st.session_state.get("selected_data_version", "Data V1")
        if selected_version not in version_files:
            selected_version = "Data V1"

        ordered_versions = [selected_version] + [
            v for v in version_files.keys() if v != selected_version
        ]

        tabs = st.tabs([f"{v} " for v in ordered_versions])
        tab_map = dict(zip(ordered_versions, tabs))

        def render_eda_for_version(version_name: str, file_path: Path):
            if not file_path.exists():
                st.error(f"❌ File for **{version_name}** not found at `{file_path}`.")
                return

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"⚠️ Failed to load {version_name}: {e}")
                return

            st.subheader(f"{version_name} Overview")
            st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.write(f"**File:** `{file_path.name}`")

            with st.expander(f"{version_name} Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            numeric_df = df.select_dtypes(include="number")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Data Types")
                dtypes_df = (
                    df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"})
                )
                st.dataframe(dtypes_df, use_container_width=True)

            with col2:
                st.markdown("#### Summary Statistics")
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for summary statistics.")

            st.markdown("---")
            st.subheader("Feature Visualizations")

            all_columns = df.columns.tolist()

            if version_name == "Data V1":
                candidates = ["sub", "trial", "profile", "concept"]
            elif version_name == "Data V2":
                candidates = ["avg_PE"]
            elif version_name == "Data V3":
                candidates = [
                    "TDNorm_avg_PE",
                    "overall_avg_PE",
                    "TDnorm_concept_learning",
                    "overall_concept_learning",
                ]
            else:
                candidates = []

            default_features = [c for c in candidates if c in all_columns]
            if not default_features:
                default_features = all_columns[:3]

            selected_features = st.multiselect(
                "Select features to visualize:",
                options=all_columns,
                default=default_features,
                help="Choose one or more columns to generate Plotly charts for.",
                key=f"feature_select_{version_name}",
            )

            if not selected_features:
                st.info("Select at least one feature to see graphs.")
            else:
                cols_in_row = 2 if len(selected_features) > 1 else 1
                row_cols = None

                for idx, col in enumerate(selected_features):
                    if idx % cols_in_row == 0:
                        row_cols = st.columns(cols_in_row)

                    target_col = row_cols[idx % cols_in_row]

                    with target_col:
                        st.markdown(f"### 📌 {col}")

                        series = df[col].dropna()
                        if series.empty:
                            st.info("No data available for this feature (all values are missing).")
                            continue

                        if pd.api.types.is_numeric_dtype(series):
                            fig = px.histogram(
                                df,
                                x=col,
                                nbins=30,
                                title=f"Distribution of {col}",
                            )
                            fig.update_layout(bargap=0.05)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            value_counts = series.value_counts().reset_index().head(30)
                            value_counts.columns = [col, "Count"]

                            fig = px.bar(
                                value_counts,
                                x=col,
                                y="Count",
                                title=f"Value Counts for {col}",
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            if not numeric_df.empty and numeric_df.shape[1] > 1:
                st.subheader("Correlation Heatmap")
                st.markdown("Checking for Data Leakage")
                corr = numeric_df.corr()
                fig = px.imshow(
                    corr,
                    text_auto=False,
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig, use_container_width=True)
            elif not numeric_df.empty:
                st.info("Only one numeric column found — correlation heatmap not meaningful.")
            else:
                st.info("No numeric columns available for correlation heatmap.")

            st.markdown("---")
            st.subheader("FSR Distribution by Target")

            FSR_COL = "FSR"
            TARGET_COL = "td_or_asd"

            LABEL_MAP = {
                0: "0: TD",
                1: "1: ASD",
                "0": "0: TD",
                "1": "1: ASD",
            }
            COLOR_MAP = {
                0: "#E74C3C",
                1: "#3498DB",
                "0": "#E74C3C",
                "1": "#3498DB",
            }

            if FSR_COL in df.columns and TARGET_COL in df.columns:
                unique_vals = df[TARGET_COL].dropna().unique()

                target_order = []
                for key in [0, 1, "0", "1"]:
                    if key in unique_vals and key not in target_order:
                        target_order.append(key)

                if not target_order:
                    target_order = list(unique_vals)

                groups = [
                    df[df[TARGET_COL] == t][FSR_COL].dropna()
                    for t in target_order
                ]
                labels = [LABEL_MAP.get(t, str(t)) for t in target_order]
                colors = [COLOR_MAP.get(t, "#7f7f7f") for t in target_order]

                fig_fsr = ff.create_distplot(
                    groups,
                    group_labels=labels,
                    show_hist=False,
                    show_rug=False,
                    colors=colors,
                    curve_type="kde",
                )

                for trace in fig_fsr.data:
                    trace.update(fill="tozeroy", opacity=0.4)

                fig_fsr.update_layout(
                    title="FSR Distribution by Target",
                    xaxis_title=FSR_COL,
                    yaxis_title="Density",
                    legend_title="Diagnosis (Target)",
                    template="simple_white",
                )

                st.plotly_chart(fig_fsr, use_container_width=True)
            else:
                st.info(f"Columns `{FSR_COL}` and/or `{TARGET_COL}` not found in this dataset.")

        for version_name, file_path in version_files.items():
            with tab_map[version_name]:
                render_eda_for_version(version_name, file_path)

    def show_results_page(self):
        st.header(f"🧮 Results — {self.model_version}")
        st.caption(f"Reading from: `Results/{self.model_version}`")

        try:
            model_folder = f"Model_{self.model_version}"
            module_name = "visualization"
            full_module_path = f"{model_folder}.{module_name}"

            viz = import_module(full_module_path)
        except Exception as e:
            st.error(f"❌ Could not import {module_name}.py for model {self.model_version}")
            st.info(f"Tried path: {full_module_path.replace('.', '/')}.py")
            st.exception(e)
            return

        visualizer = viz.ModelVisualizerInteractive()

        tab1, tab2, tab3 = st.tabs(
            ["Training Performance", "Test Performance", "Feature Visualizations"]
        )

        with tab1:
            if hasattr(self, "training_results") and self.training_results:
                st.plotly_chart(
                    visualizer.fig_cv_scores(self.training_results),
                    use_container_width=True,
                )
                st.plotly_chart(
                    visualizer.fig_overall_performance(self.training_results),
                    use_container_width=True,
                )
            else:
                st.info("Training report not found. Run train.py to generate a training report.")

        with tab2:
            if hasattr(self, "test_results") and self.test_results:
                st.plotly_chart(
                    visualizer.fig_confusion_matrix(self.test_results),
                    use_container_width=True,
                )
                st.plotly_chart(
                    visualizer.fig_classification_report(self.test_results),
                    use_container_width=True,
                )
            else:
                st.info("Test report not found. Run predict.py to generate a test report.")

        with tab3:
            if hasattr(self, "explainability_data") and self.explainability_data:
                st.plotly_chart(
                    visualizer.fig_feature_importance_by_target(self.explainability_data),
                    use_container_width=True,
                )
            else:
                st.info("Explainability report not found. Run predict.py to generate explainability outputs.")

    def show_explainability_page(self):
        st.header("🔍 Explainability (xAI)")
        st.info("Wire up SHAP/LIME visualizations here (not shown in this snippet).")

    def show_cluster_analysis_page(self):
        st.header("Cluster Analysis 🧩")
        st.caption("Unified UI for PCA+KMeans, PCA+GMM, and PCA+t-SNE+HDBSCAN")

        # ------------------------------
        # 1) Method selection
        # ------------------------------
        method_label = st.selectbox(
            "Clustering method",
            [
                "PCA + KMeans (V1)",
                "PCA + GMM (V2)",
                "PCA + t-SNE + HDBSCAN (V3)",
            ],
            index=2,
        )

        method_map = {
            "PCA + KMeans (V1)": "pca_kmeans",
            "PCA + GMM (V2)": "pca_gmm",
            "PCA + t-SNE + HDBSCAN (V3)": "tsne_hdbscan",
        }
        model_type = method_map[method_label]

        # ------------------------------
        # 1a) Participant subset selection (FSR-based)
        # ------------------------------
        subset_label = st.radio(
            "Data subset for clustering",
            [
                "All participants (full FSR range)",
                "FSR overlap region only",
                "FSR non-overlap region only",
            ],
            index=0,
            horizontal=True,
        )

        subset_map = {
            "All participants (full FSR range)": "full",
            "FSR overlap region only": "fsr_overlap",
            "FSR non-overlap region only": "fsr_nonoverlap",
        }
        subset_mode = subset_map[subset_label]

        st.caption(
            f"Current subset mode: **{subset_mode}** "
            f"(this controls how `data_preprocessed_general.csv` is generated)."
        )

        # ------------------------------
        # 2) Feature set selection (DROPDOWN)
        # ------------------------------
        feature_label = st.selectbox(
            "Feature set for clustering",
            [
                "7 Numeric (scaled)",
                "5 Numeric (scaled)",
                "5 Numeric (scaled) + NLP",
                "Without FSR (4 numeric_scaled + NLP)",
                "Custom feature subset",
            ],
            index=2,
            help="Choose a predefined feature subset, or select 'Custom feature subset' to pick features manually.",
        )

        feature_map = {
            "7 Numeric (scaled)": "all_numeric",
            "5 Numeric (scaled)": "5_numeric",
            "5 Numeric (scaled) + NLP": "5_numeric_nlp",
            "Without FSR (4 numeric_scaled + NLP)": "4_numeric_nlp",
            "Custom feature subset": "custom",
        }
        feature_mode = feature_map[feature_label]

        custom_features: list[str] | None = None

        # ------------------------------
        # 2a) Custom feature UI
        # ------------------------------
        if feature_mode == "custom":
            st.markdown("### Custom features")

            # ----- Scaled numeric features -----
            st.markdown("##### Scaled Numeric Features")
            st.info(
                "Select which **scaled numeric** features to include. "
                "You can also select any subset of NLP features below."
            )

            numeric_feature_defs = [
                ("FSR_scaled", "FSR_scaled"),
                ("BIS_scaled", "BIS_scaled"),
                ("SRS.Raw_scaled", "SRS.Raw_scaled"),
                ("TDNorm_avg_PE_scaled", "TDNorm_avg_PE_scaled"),
                ("overall_avg_PE_scaled", "overall_avg_PE_scaled"),
                ("TDnorm_concept_learning_scaled", "TDnorm_concept_learning_scaled"),
                ("overall_concept_learning_scaled", "overall_concept_learning_scaled"),
            ]

            selected_numeric = []
            num_cols = st.columns(3)
            for i, (col_name, label) in enumerate(numeric_feature_defs):
                with num_cols[i % 3]:
                    checked = st.checkbox(
                        label,
                        value=True,
                        key=f"cust_num_{col_name}",
                    )
                    if checked:
                        selected_numeric.append(col_name)

            st.markdown("---")
            st.markdown("##### NLP Features")

            st.info(
                "Select any **NLP features** you want to include. "
                "Each checkbox corresponds to a single NLP feature."
            )

            selected_nlp = []
            nlp_cols_layout = st.columns(3)
            for i, feat in enumerate(NLP_FEATURES):
                with nlp_cols_layout[i % 3]:
                    checked = st.checkbox(
                        feat,
                        value=True,
                        key=f"cust_nlp_{feat}",
                    )
                    if checked:
                        selected_nlp.append(feat)

            custom_features = selected_numeric + selected_nlp

            if not custom_features:
                st.warning(
                    "No features selected yet – clustering will fail. "
                    "Make sure to pick at least one numeric or NLP feature."
                )

            st.caption(
                f"Custom feature subset selected (`feature_mode='custom'`). "
                f"Total features: **{len(custom_features)}** "
                f"(Numeric: {len(selected_numeric)}, NLP: {len(selected_nlp)})"
            )

        # ------------------------------
        # 3) Run ID
        # ------------------------------
        run_id = st.text_input(
            "Run ID (used as suffix for saving results)",
            value="ui",
            help=(
                "If you change this, a new subfolder will be created in "
                "Results/Clustering/general/"
            ),
        )

        # Defaults for PCA / t-SNE
        pca_n_components = None  # will be set from UI for PCA models
        tsne_n_components = 2    # default → 2D t-SNE

        # ------------------------------
        # 3b) t-SNE dimensionality (for PCA + t-SNE + HDBSCAN only)
        # ------------------------------
        if model_type == "tsne_hdbscan":
            st.markdown("#### t-SNE dimensionality")
            st.caption(
                "Choose whether to embed selected PCA components into **2D** (tSNE1, tSNE2) "
                "or **3D** (tSNE1, tSNE2, tSNE3)."
            )
            tsne_dim_choice = st.radio(
                "t-SNE embedding dimensions",
                options=[2, 3],
                index=0,
                horizontal=True,
            )
            tsne_n_components = tsne_dim_choice

        st.markdown("---")

        # ==========================================================
        # 4) TWO-STAGE WORKFLOW FOR ALL PCA-BASED METHODS
        # ==========================================================
        if model_type in ("pca_kmeans", "pca_gmm", "tsne_hdbscan"):
            st.markdown("### Step 1: PCA Preview (choose components)")
            if model_type == "tsne_hdbscan":
                st.caption(
                    "First run PCA on the chosen subset + feature set. "
                    "This preview shows how many PCs are needed. In the next step, "
                    "those PCs will be used as input to t-SNE + HDBSCAN."
                )
            else:
                st.caption(
                    "First run PCA on the chosen subset + feature set. "
                    "This will generate a PCA variance plot and suggested number of PCs."
                )

            if st.button("▶ Run PCA preview"):
                with st.spinner("Running PCA preview (full pipeline, using current run ID)..."):
                    df = preprocess_clustering_data(
                        PROJECT_ROOT,
                        subset_mode=subset_mode,
                    )

                    preview_run_id = run_id

                    analyzer = GeneralClusterAnalyzer(
                        project_root=PROJECT_ROOT,
                        model_type=model_type,
                        feature_mode=feature_mode,
                        run_id=preview_run_id,
                        custom_features=custom_features,
                        pca_n_components=None,            # auto for preview
                        tsne_n_components=tsne_n_components,  # relevant for tsne_hdbscan
                    )
                    analyzer.run_pipeline(df)

                    preview_base_name = f"{model_type}_{feature_mode}_{preview_run_id}"
                    preview_base_dir = (
                        PROJECT_ROOT
                        / "Results"
                        / "Clustering"
                        / "general"
                        / preview_base_name
                    )

                    st.session_state.pca_preview_dir = str(preview_base_dir)
                    st.session_state.pca_preview_subset_mode = subset_mode
                    st.session_state.pca_preview_feature_mode = feature_mode
                    st.session_state.pca_preview_model_type = model_type

                st.success(f"✅ PCA preview completed and saved under:\n`{preview_base_dir}`")

            # --------------------------
            # Show PCA preview information if available
            # --------------------------
            preview_dir_str = st.session_state.get("pca_preview_dir", "")
            if (
                preview_dir_str
                and st.session_state.pca_preview_model_type == model_type
                and st.session_state.pca_preview_subset_mode == subset_mode
                and st.session_state.pca_preview_feature_mode == feature_mode
            ):
                preview_dir = Path(preview_dir_str)
                td_asd_dir = preview_dir / "td_asd_clusters"
                proj_dir = td_asd_dir / "projections"

                st.markdown("### PCA Preview Results")
                metrics = None
                metrics_path = td_asd_dir / "metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                    except Exception:
                        metrics = None

                var_img = proj_dir / "pca_variance.png"
                if var_img.exists():
                    st.image(str(var_img), use_container_width=True)
                else:
                    st.info("PCA variance plot not found in the preview run.")

                suggested_k = None
                max_k = None
                if metrics and "pca_params" in metrics:
                    p = metrics["pca_params"]
                    chosen_k = p.get("chosen_n_components")
                    cum_var = p.get("cum_explained_at_chosen")
                    max_k = p.get("max_n_components", None) or p.get("max_components", None)

                    if chosen_k is not None and cum_var is not None:
                        suggested_k = int(chosen_k)
                        try:
                            pct = float(cum_var) * 100.0
                            st.caption(
                                f"Preview suggests **{int(chosen_k)}** components "
                                f"(cumulative explained variance ≈ **{pct:.1f}%**)."
                            )
                        except Exception:
                            pass

                st.markdown("---")
                st.markdown("### Step 2: Select PCs and Run Clustering")

                if max_k is None:
                    max_k = 20

                options_k = list(range(2, max_k + 1))
                if suggested_k and suggested_k in options_k:
                    default_index = options_k.index(suggested_k)
                else:
                    default_index = 0

                help_text = (
                    "These components will be used as input to KMeans."
                    if model_type == "pca_kmeans"
                    else "These components will be used as input to GMM."
                    if model_type == "pca_gmm"
                    else "These components will be used as input to t-SNE + HDBSCAN."
                )

                pca_n_components = st.selectbox(
                    "Number of PCA components to use for clustering",
                    options=options_k,
                    index=default_index,
                    help=help_text,
                )

                st.caption(
                    "Once satisfied with the PCA choice, run the full clustering pipeline below."
                )

                if st.button("🚀 Run clustering (using selected PCs)"):
                    with st.spinner("Running clustering pipeline with selected PCs..."):
                        df = preprocess_clustering_data(
                            PROJECT_ROOT,
                            subset_mode=subset_mode,
                        )

                        analyzer = GeneralClusterAnalyzer(
                            project_root=PROJECT_ROOT,
                            model_type=model_type,
                            feature_mode=feature_mode,
                            run_id=run_id,
                            custom_features=custom_features,
                            pca_n_components=pca_n_components,
                            tsne_n_components=tsne_n_components,
                        )
                        analyzer.run_pipeline(df)

                        base_name = f"{model_type}_{feature_mode}_{run_id}"
                        base_dir = (
                            PROJECT_ROOT
                            / "Results"
                            / "Clustering"
                            / "general"
                            / base_name
                        )
                        st.session_state.cluster_base_dir = str(base_dir)

                    st.success(f"✅ Clustering completed and saved under:\n`{base_dir}`")
            else:
                st.info(
                    "Run the PCA preview above to see the variance explained and choose "
                    "how many principal components to use for clustering."
                )

        # ==========================================================
        # 4b) ONE-STAGE WORKFLOW FOR NON-PCA METHODS (none currently)
        # ==========================================================
        else:
            if st.button("🚀 Run clustering"):
                with st.spinner("Running clustering pipeline... this may take a few minutes."):
                    df = preprocess_clustering_data(
                        PROJECT_ROOT,
                        subset_mode=subset_mode,
                    )

                    analyzer = GeneralClusterAnalyzer(
                        project_root=PROJECT_ROOT,
                        model_type=model_type,
                        feature_mode=feature_mode,
                        run_id=run_id,
                        custom_features=custom_features,
                        pca_n_components=None,
                        tsne_n_components=tsne_n_components,
                    )
                    analyzer.run_pipeline(df)

                    base_name = f"{model_type}_{feature_mode}_{run_id}"
                    base_dir = (
                        PROJECT_ROOT
                        / "Results"
                        / "Clustering"
                        / "general"
                        / base_name
                    )
                    st.session_state.cluster_base_dir = str(base_dir)

                st.success(f"✅ Clustering completed and saved under:\n`{base_dir}`")

        # ------------------------------
        # 5) Show visuals from last run
        # ------------------------------
        base_dir_str = st.session_state.get("cluster_base_dir", "")
        if not base_dir_str:
            st.info("Run clustering above to see visualizations.")
            return

        base_dir = Path(base_dir_str)
        st.caption(f"Using clustering results from: `{base_dir}`")

        td_asd_proj_dir = base_dir / "td_asd_clusters" / "projections"
        asd_proj_dir = base_dir / "asd_subclusters" / "projections"

        if not td_asd_proj_dir.exists():
            st.warning(f"Projections directory not found: {td_asd_proj_dir}")
            return

        tab_all, tab_asd = st.tabs(
            ["All Participants (TD + ASD)", "ASD-only Participants (ASD Subclusters)"]
        )

        # --- filenames for target projection + explorer ---
        target_2d_map = {
            "pca_kmeans": "pca_target_projection_kmeans_2d.html",
            "pca_gmm": "pca_target_projection_gmm_2d.html",
            "tsne_hdbscan": "tsne_target_projection.html",
        }
        target_3d_map = {
            "pca_kmeans": "pca_target_projection_kmeans_3d.html",
            "pca_gmm": "pca_target_projection_gmm_3d.html",
            # NEW: 3D t-SNE target projection
            "tsne_hdbscan": "tsne_target_projection_3d.html",
        }

        explorer_2d_map = {
            "pca_kmeans": "pca_cluster_explorer_kmeans_2d.html",
            "pca_gmm": "pca_cluster_explorer_gmm_2d.html",
            "tsne_hdbscan": "tsne_cluster_explorer_hdbscan.html",
        }
        explorer_3d_map = {
            "pca_kmeans": "pca_cluster_explorer_kmeans_3d.html",
            "pca_gmm": "pca_cluster_explorer_gmm_3d.html",
            "tsne_hdbscan": "tsne_cluster_explorer_hdbscan_3d.html",
        }

        target_png_map = {
            "pca_kmeans": "pca_target_projection_kmeans.png",
            "pca_gmm": "pca_target_projection_gmm.png",
            "tsne_hdbscan": "tsne_target_projection.png",
        }

        def render_cluster_tab(proj_dir: Path, context_label: str):
            # ----- Load metrics (PCA + JSD info) -----
            metrics = None
            metrics_path = proj_dir.parent / "metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                except Exception:
                    metrics = None

            # ----- PCA summary + feature contributions (for all PCA-based runs) -----
            if metrics and "pca_params" in metrics:
                st.subheader(f"PCA Summary — {context_label}")

                # Variance Explained
                st.markdown("### Variance Explained")
                var_img = proj_dir / "pca_variance.png"
                if var_img.exists():
                    st.image(str(var_img), use_container_width=True)
                else:
                    st.info("PCA variance plot not found in this run.")

                p = metrics["pca_params"]
                chosen_k = p.get("chosen_n_components")
                cum_var = p.get("cum_explained_at_chosen")
                if chosen_k is not None and cum_var is not None:
                    pct = float(cum_var) * 100.0
                    st.caption(
                        f"Using **{int(chosen_k)}** PCs "
                        f"(cumulative explained variance ≈ **{pct:.1f}%**)."
                    )

                st.markdown("---")
                st.markdown("### Feature Contributions to PCs")
                st.caption(
                    "Top contributing features for PC1/PC2/PC3. "
                    "Use this to explain why clusters separate in the PCA/t-SNE plots."
                )

                feat_dir = proj_dir.parent / "feature_visuals"
                contrib_file_map = {
                    "pca_kmeans": "pca_feature_contributions_kmeans.html",
                    "pca_gmm": "pca_feature_contributions_gmm.html",
                    "tsne_hdbscan": "pca_feature_contributions_tsne_hdbscan.html",
                }
                contrib_name = contrib_file_map.get(model_type)
                contrib_html_path = feat_dir / contrib_name if contrib_name else None

                if contrib_html_path and contrib_html_path.exists():
                    html_str = contrib_html_path.read_text(encoding="utf-8")
                    components.html(html_str, height=550, scrolling=False)
                else:
                    st.info("PCA feature contribution graph not found for this run.")

            # ----- JSD visuals (all methods, with tabs) -----
            if metrics and "jsd_stats" in metrics:
                jsd_stats = metrics.get("jsd_stats", {})
                cluster_labels = jsd_stats.get("cluster_labels", [])
                per_feature = jsd_stats.get("per_feature", {})

                if cluster_labels and per_feature:
                    st.markdown("---")
                    st.subheader(f"Jensen–Shannon Divergence — {context_label}")
                    st.caption(
                        "Higher JSD ⇒ clusters differ more on that feature."
                    )

                    features_all = sorted(list(per_feature.keys()))
                    if not features_all:
                        st.info("No JSD feature data found.")
                    else:
                        tab1, tab2 = st.tabs(
                            ["🔢 Pairwise JSD Matrix", "📊 Feature JSD Summary"]
                        )

                        # -----------------------------------------
                        # TAB 1 — Pairwise JSD Matrix
                        # -----------------------------------------
                        with tab1:
                            st.subheader("Pairwise JSD (Between Clusters)")

                            default_feat = "FSR" if "FSR" in features_all else features_all[0]
                            selected_feature = st.selectbox(
                                "Select feature",
                                options=features_all,
                                index=features_all.index(default_feat),
                                key=f"jsd_matrix_feat_{context_label.replace(' ', '_')}",
                            )

                            mat = np.array(per_feature[selected_feature], dtype=float)
                            if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
                                labels = [f"C{c}" for c in cluster_labels]
                                df_mat = pd.DataFrame(
                                    mat,
                                    index=labels,
                                    columns=labels,
                                )

                                fig = px.imshow(
                                    df_mat,
                                    text_auto=".2f",
                                    color_continuous_scale="Blues",
                                    aspect="equal",
                                    title=f"JSD Matrix for Feature: {selected_feature}",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                st.write("Pairwise JSD values:")
                                st.dataframe(df_mat.style.format("{:.3f}"))
                            else:
                                st.info(
                                    f"Could not render JSD matrix for `{selected_feature}` "
                                    f"(unexpected shape: {mat.shape})."
                                )

                        # -----------------------------------------
                        # TAB 2 — Feature-Level Summary
                        # -----------------------------------------
                        with tab2:
                            st.subheader(
                                "Feature-Level JSD Score (Average Separation Power)"
                            )

                            rows = []
                            for feat_name, matrix in per_feature.items():
                                arr = np.array(matrix, dtype=float)
                                if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                                    continue
                                n = arr.shape[0]
                                if n <= 1:
                                    continue

                                mask = ~np.eye(n, dtype=bool)
                                off_diag = arr[mask]
                                if off_diag.size == 0:
                                    continue

                                mean_jsd = float(off_diag.mean())
                                rows.append({"feature": feat_name, "JSD score": mean_jsd})

                            if rows:
                                results_df = (
                                    pd.DataFrame(rows)
                                    .sort_values("JSD score", ascending=False)
                                    .reset_index(drop=True)
                                )

                                fig2 = px.bar(
                                    results_df,
                                    x="feature",
                                    y="JSD score",
                                    title="Average JSD Score Per Feature",
                                    text="JSD score",
                                )
                                fig2.update_traces(
                                    texttemplate="%{text:.3f}", textposition="outside"
                                )
                                fig2.update_layout(
                                    yaxis=dict(
                                        range=[
                                            0,
                                            results_df["JSD score"].max() * 1.2,
                                        ]
                                    )
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                                st.dataframe(
                                    results_df.style.format({"JSD score": "{:.4f}"})
                                )
                            else:
                                st.info(
                                    "No valid JSD matrices found to summarize by feature."
                                )
                else:
                    st.markdown("---")
                    st.subheader(f"Jensen–Shannon Divergence — {context_label}")
                    st.info("`jsd_stats` found in metrics, but no usable data to visualize.")

            # ----- Cluster explorer (2D / 3D) -----
            st.markdown("---")
            st.subheader(f"Cluster Explorer — {context_label}")
            st.caption(
                "Use the legend to toggle clusters. The panel on the right updates with "
                "cluster-wise stats (TD/ASD counts + numeric feature summaries)."
            )

            explorer_2d = proj_dir / explorer_2d_map[model_type]
            explorer_3d_name = explorer_3d_map.get(model_type)
            explorer_3d = proj_dir / explorer_3d_name if explorer_3d_name else None

            has_explorer_2d = explorer_2d.exists()
            has_explorer_3d = explorer_3d is not None and explorer_3d.exists()

            if not has_explorer_2d and not has_explorer_3d:
                st.info("Cluster explorer HTML not found for this run.")
            else:
                modes = []
                path_by_mode = {}
                if has_explorer_2d:
                    modes.append("2D")
                    path_by_mode["2D"] = explorer_2d
                if has_explorer_3d:
                    modes.append("3D")
                    path_by_mode["3D"] = explorer_3d

                if len(modes) == 1:
                    if "3D" in modes:
                        if model_type in ("pca_kmeans", "pca_gmm"):
                            st.caption(
                                "Only **3D** view available for this run.\n"
                                "- PCA models: PC1 vs PC2 vs PC3"
                            )
                        else:
                            st.caption(
                                "Only **3D** t-SNE view available for this run "
                                "(tSNE1 vs tSNE2 vs tSNE3)."
                            )
                    else:
                        if model_type in ("pca_kmeans", "pca_gmm"):
                            st.caption(
                                "Only **2D** view available for this run "
                                "(PC1 vs PC2; fewer than 3 usable principal components "
                                "or 3D file was not generated)."
                            )
                        else:
                            st.caption(
                                "Only **2D** t-SNE view available for this run "
                                "(tSNE1 vs tSNE2)."
                            )

                    html_str = path_by_mode[modes[0]].read_text(encoding="utf-8")
                    components.html(html_str, height=650, scrolling=True)

                else:
                    if model_type in ("pca_kmeans", "pca_gmm"):
                        st.caption(
                            "Use the selector below to switch between "
                            "**2D (PC1 vs PC2)** and **3D (PC1 vs PC2 vs PC3)** views."
                        )
                    elif model_type == "tsne_hdbscan":
                        st.caption(
                            "Use the selector below to switch between "
                            "**2D (tSNE1 vs tSNE2)** and **3D (tSNE1 vs tSNE2 vs tSNE3)** views."
                        )
                    else:
                        st.caption(
                            "Use the selector below to switch between **2D** and **3D** views."
                        )

                    default_mode = "3D" if "3D" in modes else "2D"
                    default_index = modes.index(default_mode)
                    explorer_mode = st.selectbox(
                        "Cluster explorer view",
                        options=modes,
                        index=default_index,
                        key=f"cluster_explorer_mode_{context_label.replace(' ', '_')}",
                    )
                    html_str = path_by_mode[explorer_mode].read_text(encoding="utf-8")
                    components.html(html_str, height=650, scrolling=True)

            st.markdown("---")
            st.subheader(f"Target Projection — {context_label}")

            # ----- Target projection (2D / 3D) -----
            target_2d = proj_dir / target_2d_map[model_type]

            # robust 3D detection
            target_3d = None
            if model_type in ("pca_kmeans", "pca_gmm"):
                t3_name = target_3d_map.get(model_type)
                if t3_name:
                    candidate = proj_dir / t3_name
                    if candidate.exists():
                        target_3d = candidate
            elif model_type == "tsne_hdbscan":
                # Look for common 3D t-SNE target filenames
                for name in ["tsne_target_projection3d.html", "tsne_target_projection_3d.html"]:
                    candidate = proj_dir / name
                    if candidate.exists():
                        target_3d = candidate
                        break

            has_2d = target_2d.exists()
            has_3d = target_3d is not None and target_3d.exists()

            if has_2d or has_3d:
                modes = []
                path_by_mode = {}
                if has_2d:
                    modes.append("2D")
                    path_by_mode["2D"] = target_2d
                if has_3d:
                    modes.append("3D")
                    path_by_mode["3D"] = target_3d

                if len(modes) == 1:
                    if "3D" in modes:
                        if model_type in ("pca_kmeans", "pca_gmm"):
                            st.caption(
                                "Only **3D** target projection available "
                                "(PC1 vs PC2 vs PC3; at least 3 principal components)."
                            )
                        else:
                            st.caption(
                                "Only **3D** t-SNE target projection available "
                                "(tSNE1 vs tSNE2 vs tSNE3)."
                            )
                    else:
                        if model_type in ("pca_kmeans", "pca_gmm"):
                            st.caption(
                                "Only **2D** target projection available "
                                "(PC1 vs PC2; fewer than 3 usable principal components "
                                "or 3D file was not generated)."
                            )
                        else:
                            st.caption(
                                "Only **2D** t-SNE target projection available "
                                "(tSNE1 vs tSNE2)."
                            )

                    html_str = path_by_mode[modes[0]].read_text(encoding="utf-8")
                    components.html(html_str, height=600, scrolling=True)

                else:
                    if model_type in ("pca_kmeans", "pca_gmm"):
                        st.caption(
                            "Switch between **2D (PC1 vs PC2)** and "
                            "**3D (PC1 vs PC2 vs PC3)** target views."
                        )
                    elif model_type == "tsne_hdbscan":
                        st.caption(
                            "Switch between **2D (tSNE1 vs tSNE2)** and "
                            "**3D (tSNE1 vs tSNE2 vs tSNE3)** target views."
                        )
                    else:
                        st.caption(
                            "Switch between **2D** and **3D** target views."
                        )

                    default_mode = "3D" if "3D" in modes else "2D"
                    default_index = modes.index(default_mode)
                    viz_mode = st.selectbox(
                        "Visualization mode (target projection)",
                        options=modes,
                        index=default_index,
                        key=f"target_viz_mode_{context_label.replace(' ', '_')}",
                    )

                    html_str = path_by_mode[viz_mode].read_text(encoding="utf-8")
                    components.html(
                        html_str,
                        height=650 if viz_mode == "3D" else 600,
                        scrolling=True,
                    )

            else:
                # No HTML → fallback to PNG if available
                if target_2d.exists():
                    if model_type == "tsne_hdbscan":
                        st.caption("t-SNE target projection is available in **2D** only.")
                    html_str = target_2d.read_text(encoding="utf-8")
                    components.html(html_str, height=600, scrolling=True)
                else:
                    target_png = proj_dir / target_png_map[model_type]
                    if target_png.exists():
                        if model_type == "tsne_hdbscan":
                            st.caption("t-SNE target projection is available in **2D** only.")
                        st.image(str(target_png), use_container_width=True)
                    else:
                        st.info(
                            f"Target projection file not found: "
                            f"`{target_2d.name}` or `{target_png.name}`"
                        )

        with tab_all:
            render_cluster_tab(td_asd_proj_dir, "All Participants")

        with tab_asd:
            if asd_proj_dir.exists():
                render_cluster_tab(asd_proj_dir, "ASD-only")
            else:
                st.info("No ASD-only subclustering folder found (maybe no ASD rows).")




def main():
    app = DemoApp()

    st.sidebar.title("Navigation")

    pages = {
        "🏠 Home": "home",
        "📈 Data & Model": "overview",
        "📊 EDA": "eda",
        "🧮 Model Results": "results",
        "🔍 xAI - Explainability": "explainability",
        "🧩 Cluster Analysis": "clustering",
    }

    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_key

    st.sidebar.markdown("---")
    st.sidebar.markdown("###### Note: These fields update when new page is selected.")

    st.sidebar.markdown("### Data Version (current)")
    st.sidebar.info(st.session_state.data_version)

    st.sidebar.markdown("### Model Version (current)")
    st.sidebar.info(st.session_state.model_version)

    st.sidebar.markdown("### Model Info")
    st.sidebar.caption(get_model_info(st.session_state.model_version))

    # Ensure artifacts exist for selected version
    app.load_results()

    # Route to selected page
    if st.session_state.page == "home":
        app.show_home_page()
    elif st.session_state.page == "overview":
        app.show_model_and_data_page()
    elif st.session_state.page == "eda":
        app.show_eda_page()
    elif st.session_state.page == "results":
        app.show_results_page()
    elif st.session_state.page == "explainability":
        app.show_explainability_page()
    elif st.session_state.page == "clustering":
        app.show_cluster_analysis_page()


if __name__ == "__main__":
    main()
