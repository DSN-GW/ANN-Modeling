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
import math



# ----- App / Layout -----
st.set_page_config(
    page_title="TD/ASD Classification - Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Paths -----
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
CLUSTER_RESULTS_ROOT = RESULTS_ROOT / "Clustering"
DATA_ROOT = PROJECT_ROOT / "data"
# Ensure Root/src is on sys.path so Model_Vx, etc. can be imported
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# print(PROJECT_ROOT)
# print(RESULTS_ROOT)
# print(DATA_ROOT)
# print(SRC_ROOT)


def _discover_models() -> list[str]:
    """Return available model versions by scanning Results/* that look like V{number}."""
    if not RESULTS_ROOT.exists():
        return ["V1", "V2", "V3", "V4", "V5"]
    versions = []
    for p in sorted(RESULTS_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("V") and p.name[1].isdigit():
            versions.append(p.name.upper())
    return versions or ["V1", "V2", "V3", "V4", "V5"]

def _discover_clustering_models() -> list[str]:
    """Return available clustering model versions by scanning Results/* that look like Clustering_V{number}."""
    if not CLUSTER_RESULTS_ROOT.exists():
        return ["No Clustering Results Found"]
    versions = []
    for p in sorted(CLUSTER_RESULTS_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("V") and p.name[1].isdigit():
            versions.append(p.name.upper())
    return versions or ["No Clustering Results Found"]

def _discover_datasets() -> list[str]:
    """Return available datasets by scanning data/* that look like Data_v{number}."""
    if not DATA_ROOT.exists():
        return ["V1", "V2", "V3",]
    versions = []
    for p in sorted(DATA_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("DATA_V"):
            versions.append(p.name.upper())
    return versions or ["V1", "V2", "V3", "V4"]

# Initialize session state and model selection
if "page" not in st.session_state:
    st.session_state.page = "home"

if "model_version" not in st.session_state:
    st.session_state.model_version = "V7-1"

if "data_version" not in st.session_state:
    st.session_state.data_version = "DATA_V7-1"

if "available_models" not in st.session_state:
    st.session_state.available_models = _discover_models()

if "available_datasets" not in st.session_state:
    st.session_state.available_datasets = _discover_datasets()

if "clustering_version" not in st.session_state:
    st.session_state.clustering_version = "V1"

if "available_clustering_models" not in st.session_state:
    st.session_state.available_clustering_models = _discover_clustering_models()

class DemoApp:
    def __init__(self):
        self.model_version = st.session_state.model_version
        self.data_version = st.session_state.data_version
        self.clustering_version = st.session_state.clustering_version
        self.results_dir = RESULTS_ROOT / self.model_version
        self.data_dir = DATA_ROOT / f"Data_{self.model_version}"
        self.viz_dir = self.results_dir / "visualizations"
        self.predictor = None
        self.training_results = None
        self.test_results = None
        self.explainability_data = None

    # ----- Model Version Selection -----
    def set_model_version(self, version: str):
        """Update the app to a new model version and refresh result paths."""
        self.model_version = version
        st.session_state.model_version = version
        self.results_dir = RESULTS_ROOT / version
        self.data_dir = DATA_ROOT / f"Data_{self.model_version}"
        self.viz_dir = self.results_dir / "visualizations"

    def set_data_version(self, version: str):
        """Update the app to a new data version and refresh paths."""
        self.data_version = version
        st.session_state.data_version = version
        self.data_dir = DATA_ROOT / self.data_version

    def set_clustering_version(self, version: str):
        """Update the app to a new clustering version and refresh paths."""
        self.clustering_version = version
        st.session_state.clustering_version = version



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
            except:
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
        st.header(f"🏠 About the Project")

        # =======================
        # Project Overview
        # =======================

        st.markdown("""
            ### Project Overview
            This project aims to investigate how **interpretable,
            multi-modal machine learning models** can enhance the
            diagnosis and understand of **Autism Spectrum Disorder
            (ASD)**. By integrating **behavioral** and **linguistic
            data**, the study aims to move beyond traditional, subjective,
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

        # ==========================================
        # 🎯 Key Objectives | 🧰 Tools & Techniques
        # ==========================================

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

        # =======================
        # 🧭 App Navigation Guide
        # =======================
        st.markdown("---")
        st.subheader("🧭 App Navigation Guide")

        st.markdown("""
        Use the tabs on the left to explore different parts of the project:

        - **🏠 Home** – Project overview, Key objectives, Tools and techniques
        - **📈 Data & Model** – Data versions, Data dictionary, Models selection
        - **📊 EDA** - Exploratory Data Analysis for the chosen features
        - **🧮 Model Results** – Train accuracy, Test accuracy, Confusion matrix, F1-scores
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


        with tab1:
            # ---- Data selector ----
            st.markdown("## Data Overview")

            st.markdown("""
            The data is provided by The George Washington University - Department of Psychological 
            & Brain Sciences. There are 1,119 participants in this data aging from 8 to 12 years old.
            """)

            # File paths

            dict_v1_path = DATA_ROOT /  "Data_card" / "LLM data_trial_dictionary.csv"
            dict_v2_path = DATA_ROOT /  "Data_card" / "LLM data dictionary.csv"
            dict_v3_path = DATA_ROOT /  "Data_card" / "LLM data_aggregate_dictionary.csv"

            data_v1_path = DATA_ROOT /  "Data_card" / "LLM data_trial.csv"
            data_v2_path = DATA_ROOT /  "Data_card" / "LLM data.csv"
            data_v3_path = DATA_ROOT /  "Data_card" / "LLM data_aggregate.csv"


            # --- Data V1 ---
            with st.expander("Data V1: Raw Trial-Level Data", expanded=False):
                st.write("""
                    - 187,187 observations  
                    - 19 variables  
                    - Contains raw data at the trial level for all participants  
                    """)

                # Data dictionary for V1
                st.markdown("#### Data Dictionary – Data V1")
                if not dict_v1_path.exists():
                    st.error("❌ Data V1 dictionary file not found.")
                else:
                    try:
                        dict_v1 = pd.read_csv(dict_v1_path)
                        st.dataframe(dict_v1, use_container_width=True, height=300)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V1 dictionary: {e}")

                # Data V1 Preview
                st.markdown("#### Data Preview")
                if not data_v1_path.exists():
                    st.error("❌ Data V1 file not found. Check the path or file name.")
                else:
                    try:
                        df_v1 = pd.read_csv(data_v1_path)
                        st.dataframe(df_v1.head(20))
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V1: {e}")

                    # # Buttons
                    # col1, col2 = st.columns(2)
                    # with col1:
                    #     if st.button("Use Data V1 for Modeling", key="use_v1"):
                    #         self.set_data_version("Data V1")
                    #         st.success(f"✅ Data version set to **{self.data_version}**.")
                    #         st.rerun()
                    # with col2:
                    #     if st.button("See EDA for Data V1", key="eda_v1"):
                    #         st.session_state["page"] = "eda"
                    #         st.session_state["selected_data_version"] = "Data V1"
                    #         st.rerun()

            # --- Data V2 ---
            with st.expander("Data V2: Aggregated with Slope Features", expanded=False):
                st.write("""
                    - 2,648 observations  
                    - 11 variables  
                    - Aggregated per participant for each profile with computed slope features (avg_PE) 
                    """)

                # Data dictionary for V2
                st.markdown("#### Data Dictionary – Data V2")
                if not dict_v2_path.exists():
                    st.error("❌ Data V2 dictionary file not found.")
                else:
                    try:
                        dict_v2 = pd.read_csv(dict_v2_path)
                        st.dataframe(dict_v2, use_container_width=True, height=300)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V2 dictionary: {e}")

                # Data V2 Preview
                st.markdown("#### Data Preview")
                if not data_v2_path.exists():
                    st.error("❌ Data V2 file not found. Check the path or file name.")
                else:
                    try:
                        df_v2 = pd.read_csv(data_v2_path)
                        st.dataframe(df_v2.head(20))
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V2: {e}")

                # # Buttons
                # col1, col2 = st.columns(2)
                # with col1:
                #     if st.button("Use Data V2 for Modeling", key="use_v2"):
                #         self.set_data_version("Data V2")
                #         st.success(f"✅ Data version set to **{self.data_version}**.")
                #         st.rerun()
                # with col2:
                #     if st.button("See EDA for Data V2", key="eda_v2"):
                #         st.session_state["page"] = "eda"
                #         st.session_state["selected_data_version"] = "Data V2"
                #         st.rerun()

            # --- Data V3 ---
            with st.expander("Data V3: Aggregated with Concept Learning Features", expanded=False):
                st.write("""
                    - 1,119 observations  
                    - 10 variables  
                    - Participant-level dataset with cognitive and behavioral learning features  
                    """)

                # Data dictionary for V2
                st.markdown("#### Data Dictionary – Data V3")
                if not dict_v3_path.exists():
                    st.error("❌ Data V3 dictionary file not found.")
                else:
                    try:
                        dict_v3 = pd.read_csv(dict_v3_path)
                        st.dataframe(dict_v3, use_container_width=True, height=300)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V3 dictionary: {e}")

                # Data V3 Preview
                st.markdown("#### Data Preview")
                if not data_v3_path.exists():
                    st.error("❌ Data V3 file not found. Check the path or file name.")
                else:
                    try:
                        df_v3 = pd.read_csv(data_v3_path)
                        st.dataframe(df_v3.head(20))
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Data V3: {e}")

                # # Buttons
                # col1, col2 = st.columns(2)
                # with col1:
                #     if st.button("Use Data V3 for Modeling", key="use_v3"):
                #         self.set_data_version("Data V3")
                #         st.success(f"✅ Data version set to **{self.data_version}**.")
                #         st.rerun()
                # with col2:
                #     if st.button("See EDA for Data V3", key="eda_v3"):
                #         st.session_state["page"] = "eda"
                #         st.session_state["selected_data_version"] = "Data V3"
                #         st.rerun()

        with tab2:
            # # --- Model V1 ---
            # with st.expander("Model V1", expanded=False):
            #     st.write("""
            #                 - Base Dataset : Data V2
            #                 - Engineered Features:
            #                     - PE (as avg_PE)
            #                     - NLP Features:
            #                 - Train-Test Split : 80-20 Stratified 5-Fold CV
            #                 -
            #                 """)
            #
            # # --- Model V2 ---
            # with st.expander("Model V2", expanded=False):
            #     st.write("""
            #                 - Base Dataset : Data V2
            #                 - Engineered Features:
            #                     - PE (as avg_PE)
            #                     - NLP Features:
            #                     - Characteristic Features:
            #                         -LLM Agent : Claud Sonnet 3.5 via Bedrock API
            #                 - Train-Test Split : 80-20 Stratified 5-Fold CV
            #                 -
            #                 """)
            #
            # # --- Model V3 ---
            # with st.expander("Model V3", expanded=False):
            #     st.write("""
            #                 - Base Dataset : Data V2
            #                 - Engineered Features:
            #                     - PE (as avg_PE)
            #                     - NLP Features:
            #                     - Characteristic Features
            #                 - Train-Test Split : 80-20 Stratified 5-Fold CV
            #                 -
            #                 """)

            # --- Model V7-1 ---
            with st.expander("Model V7-1", expanded=False):
                st.write("""
                            - Base Dataset : Data V3 
                            - Features Used : FSR + PE + Concept Learning + NLP Features
                            - Train-Test Split : 80-20 Stratified 5-Fold CV
                            - Model : XGBOOST
                            - Metric Maximized : Accuracy
                            - Hyperparameters : *Need to be Filled in*
                            - Intuition/Reasoning for Model - Developed New Concept Features to Replace Previous Slope Calculation
                            """)

            # --- Model V7-2 ---
            with st.expander("Model V7-2", expanded=False):
                st.write("""
                            - Base Dataset : Data V3 
                            - Features Used : FSR + PE + Concept Learning + NLP Features
                            - Train-Test Split : FSR Overlap Region as Test Set
                            - Model : XGBOOST
                            - Metric Maximized : Accuracy
                            - Hyperparameters : *Need to be Filled in*
                            - Intuition/Reasoning for Model - Developed New Concept Features to Replace Previous Slope Calculation
                            """)

            # --- Model V7-3 ---
            with st.expander("Model V7-3", expanded=False):
                st.write("""
                            - Base Dataset : Data V3 
                            - Features Used : FSR + PE + Concept Learning + NLP Features
                            - Train-Test Split : FSR Overlap Region as Train Set
                            - Model : XGBOOST
                            - Metric Maximized : Accuracy
                            - Hyperparameters : *Need to be Filled in*
                            - Intuition/Reasoning for Model - Developed New Concept Features to Replace Previous Slope Calculation
                            """)

            # --- Model V8-1 ---
            with st.expander("Model V8-1", expanded=False):
                st.write("""
                            - Base Dataset : Data V3 
                            - Features Used : PE + Concept Learning + NLP Features
                            - Train-Test Split : 80-20 Stratified 5-Fold CV
                            - Model : XGBOOST
                            - Metric Maximized : Accuracy
                            - Hyperparameters : *Need to be Filled in*
                            - Intuition/Reasoning for Model - Developed Non FSR model to compare against V7-1
                            """)

            # --- Model V8-2 ---
            with st.expander("Model V8-2", expanded=False):
                st.write("""
                            - Base Dataset : Data V3 
                            - Features Used : PE + Concept Learning + NLP Features
                            - Train-Test Split : FSR Overlap Region as Test Set
                            - Model : XGBOOST
                            - Metric Maximized : Accuracy
                            - Hyperparameters : *Need to be Filled in*
                            - Intuition/Reasoning for Model - Developed Non FSR model to compare against V7-2
                            """)

            # --- Model V8-3 ---
            with st.expander("Model V8-3", expanded=False):
                st.write("""
                            - Base Dataset : Data V3 
                            - Features Used : PE + Concept Learning + NLP Features
                            - Train-Test Split : FSR Overlap Region as Train Set
                            - Model : XGBOOST
                            - Metric Maximized : Accuracy
                            - Hyperparameters : *Need to be Filled in*
                            - Intuition/Reasoning for Model - Developed Non FSR model to compare against V7-3
                            """)

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

        # ---------- 1. Which data version? (default = Data V1) ----------

        data_dir = DATA_ROOT /  "Data_card"

        version_files = {
            "Data V1": data_dir / "LLM data_trial.csv",
            "Data V2": data_dir / "LLM data.csv",
            "Data V3": data_dir / "LLM data_aggregate.csv",
        }

        selected_version = st.session_state.get("selected_data_version", "Data V1")
        if selected_version not in version_files:
            selected_version = "Data V1"

        # Put the selected version FIRST so its tab is active by default
        ordered_versions = [selected_version] + [
            v for v in version_files.keys() if v != selected_version
        ]

        tabs = st.tabs([f"{v} " for v in ordered_versions])
        tab_map = dict(zip(ordered_versions, tabs))

        # ---------- Helper: render EDA for a single version ----------
        def render_eda_for_version(version_name: str, file_path: Path):
            if not file_path.exists():
                st.error(f"❌ File for **{version_name}** not found at `{file_path}`.")
                return

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"⚠️ Failed to load {version_name}: {e}")
                return

            # === Overview ===
            st.subheader(f"{version_name} Overview")
            st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.write(f"**File:** `{file_path.name}`")

            # === Data Preview (expander) ===

            with st.expander(f"{version_name} Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # === Data types + summary stats side-by-side ===
            numeric_df = df.select_dtypes(include="number")
            cat_df = df.select_dtypes(exclude="number")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Data Types")
                dtypes_df = (
                    df.dtypes
                    .reset_index()
                    .rename(columns={"index": "Column", 0: "Type"})
                )
                st.dataframe(dtypes_df, use_container_width=True)

            with col2:
                st.markdown("#### Summary Statistics")
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for summary statistics.")



            # === Feature-wise visualizations ===
            st.markdown("---")
            st.subheader("Feature Visualizations")

            all_columns = df.columns.tolist()

            # Choose default features based on which data version we're in
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

            # Keep only those that actually exist in the dataframe
            default_features = [c for c in candidates if c in all_columns]

            # Fallback in case some names don't exist
            if not default_features:
                default_features = all_columns[:3]

            selected_features = st.multiselect(
                "Select features to visualize:",
                options=all_columns,
                default=default_features,
                help="Choose one or more columns to generate Plotly charts for.",
                key=f"feature_select_{version_name}",  # keep per-tab state
            )

            if not selected_features:
                st.info("Select at least one feature to see graphs.")
                return

            cols_in_row = 2 if len(selected_features) > 1 else 1
            row_cols = None

            for idx, col in enumerate(selected_features):
                # New row every `cols_in_row` features
                if idx % cols_in_row == 0:
                    row_cols = st.columns(cols_in_row)

                target_col = row_cols[idx % cols_in_row]

                with target_col:
                    st.markdown(f"### 📌 {col}")

                    series = df[col].dropna()
                    if series.empty:
                        st.info("No data available for this feature (all values are missing).")
                        continue

                    n_unique = series.nunique()

                    # Numeric → histogram
                    if pd.api.types.is_numeric_dtype(series):
                        fig = px.histogram(
                            df,
                            x=col,
                            nbins=30,
                            title=f"Distribution of {col} ",
                            subtitle=f"Unique values: {n_unique}",
                        )
                        fig.update_layout(bargap=0.05)
                        st.plotly_chart(fig, use_container_width=True)

                    # Categorical → value-counts bar chart
                    else:
                        value_counts = series.value_counts().reset_index().head(30)
                        value_counts.columns = [col, "Count"]

                        fig = px.bar(
                            value_counts,
                            x=col,
                            y="Count",
                            title=f"Value Counts for {col} ",
                            subtitle= f"Unique values: {n_unique}",
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)



                # === Correlation heatmap ===
            st.markdown("---")

            if not numeric_df.empty and numeric_df.shape[1] > 1:
                st.subheader("Correlation Heatmap")
                st.markdown("Checking for Data Leakage")
                corr = numeric_df.corr()
                fig = px.imshow(
                    corr,
                    text_auto=False,
                    color_continuous_scale='RdBu',
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
            elif not numeric_df.empty:
                st.info("Only one numeric column found — correlation heatmap not meaningful.")
            else:
                st.info("No numeric columns available for correlation heatmap.")

            # === FSR distribution by target ===
            st.markdown("---")
            st.subheader("FSR Distribution by Target")

            FSR_COL = "FSR"
            TARGET_COL = "td_or_asd"

            # Fixed label + color mapping (reuse this everywhere in your app)
            LABEL_MAP = {
                0: "0: TD",
                1: "1: ASD",
                "0": "0: TD",
                "1": "1: ASD",
            }
            COLOR_MAP = {
                0: "#E74C3C",  # TD = red
                1: "#3498DB",  # ASD = blue
                "0": "#E74C3C",
                "1": "#3498DB",
            }

            if FSR_COL in df.columns and TARGET_COL in df.columns:
                # Get unique target values present in this df
                unique_vals = df[TARGET_COL].dropna().unique()

                # Enforce a consistent order: 0 then 1 (works if stored as int or str)
                target_order = []
                for key in [0, 1, "0", "1"]:
                    if key in unique_vals and key not in target_order:
                        target_order.append(key)

                # Fallback in case something unexpected is in the column
                if not target_order:
                    target_order = list(unique_vals)

                groups = [
                    df[df[TARGET_COL] == t][FSR_COL].dropna()
                    for t in target_order
                ]
                labels = [LABEL_MAP.get(t, str(t)) for t in target_order]
                colors = [COLOR_MAP.get(t, "#7f7f7f") for t in target_order]  # default grey if unknown

                # Create smoothed density plot (KDE)
                fig_fsr = ff.create_distplot(
                    groups,
                    group_labels=labels,
                    show_hist=False,
                    show_rug=False,
                    colors=colors,
                    curve_type="kde",
                )

                # Fill under each curve for smooth shaded look
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

        # ---------- Render each tab's EDA ----------
        for version_name, file_path in version_files.items():
            with tab_map[version_name]:
                render_eda_for_version(version_name, file_path)

    def show_results_page(self):
        st.header(f"🧮 Results")
        st.caption(f"Select a Model to View its results")
        # TODO: load and display figures/metrics specific to self.model_version
        # st.info("Results page ")

        st.markdown("""
                    <style>
                    /* Center and evenly space Streamlit tabs */
                    div[data-baseweb="tab-list"] {
                        justify-content: space-evenly;
                    }
                    </style>
                """, unsafe_allow_html=True)

        # ---- Model selector ----
        st.markdown("#### Select a Model Version")
        selected = st.selectbox(
            "Choose which model version to explore:",
            options=st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.model_version)
            if st.session_state.model_version in st.session_state.available_models else 0,
            key="model_version_select"
        )

        # Apply selection immediately if changed
        if selected != self.model_version:
            self.set_model_version(selected)
            st.success(f"Model version set to **{self.model_version}**.")
            st.rerun()

        # ---- Dynamic import of explainers ----
        try:
            # Import path: Model_Vx.shap_explain  or  Model_Vx.lime_explain
            model_folder = f"Model_{self.model_version}"
            module_name = "visualization"
            full_module_path = f"{model_folder}.{module_name}"

            viz = import_module(full_module_path)
            # st.success(f"Loaded {full_module_path}")
        except Exception as e:
            st.error(f"❌ Could not import {module_name}.py for model {self.model_version}")
            st.info(f"Tried path: {full_module_path.replace('.', '/')}.py")
            st.exception(e)
            return

        visualizer = viz.ModelVisualizerInteractive()

        tab1, tab2, = st.tabs(["Test Performance", "Feature Visualizations"])

        with tab1:
            if self.test_results:
                st.plotly_chart(visualizer.fig_confusion_matrix(self.test_results), use_container_width=True)

                st.plotly_chart(visualizer.fig_classification_report(self.test_results), use_container_width=True)

                st.plotly_chart(visualizer.fig_auc_roc_test(), use_container_width=True)

            else:
                st.info("Test Report not found. Run predict.py to generate a test report.")

        with tab2:
            if self.explainability_data:
                st.plotly_chart(visualizer.fig_feature_importance_by_target(self.explainability_data), use_container_width=True)

            else:
                st.info("Test Report not found. Run predict.py to generate a test report.")

    def show_explainability_page(self):
        st.header(f"🔍 Explainability — {self.model_version}")
        st.caption(f"Using artifacts from: `Results/{self.model_version}`")
        # TODO: show SHAP/LIME global & local explanations tied to self.model_version

        st.markdown("""
                    <style>
                    /* Center and evenly space Streamlit tabs */
                    div[data-baseweb="tab-list"] {
                        justify-content: space-evenly;
                    }
                    </style>
                """, unsafe_allow_html=True)

        tab1, tab2, = st.tabs(["LIME", "SHAP"])

        def build_lime_explanation_text(result, top_k=7):

            pred = result["predicted_class"]
            proba = result["prediction_proba"]

            cls_name = "ASD" if pred == 1 else "TD"
            cls_proba = proba[pred]
            other_proba = proba[1 - pred]
            other_cls = "TD" if pred == 1 else "ASD"

            weights = result["weights"]

            # Sort weights by absolute influence
            top = sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:top_k]

            positives = [(f, w) for f, w in top if w > 0]
            negatives = [(f, w) for f, w in top if w < 0]

            txt_pos = "\n".join([f"- **{f}** (+{w:.3f})" for f, w in positives]) or "*None*"
            txt_neg = "\n".join([f"- **{f}** ({w:.3f})" for f, w in negatives]) or "*None*"

            return f"""
        # 📘 Local Explanation Summary

         \n- *Prediction:* *{cls_name}* 
         \n- *Confidence:* {cls_proba:.2f}  
         
        \n### Features pushing **toward {cls_name}**
        \n{txt_pos}

        \n### Features pushing **away from {cls_name}**
        \n{txt_neg}

        \n### 💡 How to read the plot
        - Orange bars push the prediction toward ASD  
        - Blue bars push the prediction toward TD  
        - The longer the bar, the stronger the influence  
        - Each feature label contains the bin/interval your value falls into  
        - LIME explains a single prediction by approximating the model locally with a simple linear model  
            """

        def build_shap_explanation_text(result, top_k=5):
            pred = result["predicted_class"]
            proba = result["prediction_proba"]
            shap_vals = np.array(result["shap_values"])
            feature_vals = result["feature_values"]
            base_value = result["base_value"]

            cls_name = "ASD" if pred == 1 else "TD"
            other_cls = "TD" if pred == 1 else "ASD"
            cls_proba = proba[pred]
            other_proba = proba[1 - pred]

            # Sort top-k contributions
            indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]
            top_feats = [(list(feature_vals.keys())[i], shap_vals[i]) for i in indices]

            pos = [(f, v) for f, v in top_feats if v > 0]
            neg = [(f, v) for f, v in top_feats if v < 0]

            txt_pos = "\n".join([f"- **{f}** (+{v:.3f})" for f, v in pos]) or "*None*"
            txt_neg = "\n".join([f"- **{f}** ({v:.3f})" for f, v in neg]) or "*None*"

            return f"""
            \n# 📘 SHAP Local Explanation Summary (Row {result['index']})
            \n*Prediction: {cls_name}  
            \n*Confidence: `{cls_proba:.2f}`  
            
            \n### Features pushing **toward ASD**
            \nThese features *increased* the model's confidence in Selecting ASD Label:
            \n{txt_pos}
 
            
            \n### Features pushing **toward TD**
            \nThese features pulled the prediction *away* from the ASD Label:
            \n{txt_neg}
            
            ---
            
            \n### 📘 How to interpret this SHAP plot
            
            \n- **Longer bars = stronger influence**
            \n- **Blue bars**: push prediction **toward TD**
            \n- **Red bars**: push prediction **toward ASD**
            \n- The SHAP waterfall plot visualizes how the model moved from the baseline
                prediction *E[f(x)] = {base_value:.3f}* to the final output *f(x)* by
                adding positive and negative feature contributions.
            \n- The final score is then converted into class probabilities `{proba}`
            
        """

        with tab1:
            # ---- Dynamic import of explainers ----
            try:
                model_folder = f"Model_{self.model_version}"
                lime_module_name = "lime_explain"
                full_module_path = f"{model_folder}.{lime_module_name}"

                lime_expl = import_module(full_module_path)
                st.success(f"Loaded {full_module_path}")
            except Exception as e:
                st.error(f"❌ Could not import {lime_module_name}.py for model {self.model_version}")
                st.info(f"Tried path: {full_module_path.replace('.', '/')}.py")
                st.exception(e)
                return

            with st.expander("Global Explainability", expanded=False):
                st.write("""
                    
                    """)

                with st.spinner(f"Running global explainability..."):
                    results = lime_expl.explain_global()
                st.success("Global explainability generated.")
                for output in results.get("outputs", []):
                    if output.endswith(".png"):
                        st.image(output, caption=f"LIME Global Plot")
                    elif output.endswith(".json"):
                        st.markdown("**Top global features:**")
                        with open(output, "r") as f:
                            st.json(dict(list(json.load(f).items())[:10]))

            with st.expander("Local Explainability", expanded=False):
                st.write("""
        
                    """)

                train_data_path = self.data_dir / f"LLM_data_train_preprocessed_{self.model_version}.csv"
                test_data_path = self.data_dir / f"LLM_data_test_preprocessed_{self.model_version}.csv"

                train_data = pd.read_csv(train_data_path)
                test_data = pd.read_csv(test_data_path)

                data_selector = st.radio(
                    "Select Data Source",
                    [
                        "Train",
                        "Test",
                    ],
                    index=1,
                    horizontal=True,
                )

                # overall_data = pd.concat([train_data, test_data])
                # overall_data.reset_index(inplace=True)
                # st.dataframe(overall_data.head(10))

                if data_selector == "Train":
                    overall_data = train_data
                else:
                    overall_data = test_data


                row_idx = st.number_input("Select a row index to explain", min_value=0, max_value=overall_data.shape[0], step=1, value=0, key="LIME_local_select")

                with st.expander("View Selected Index", expanded=False):
                    st.dataframe(overall_data.iloc[row_idx])
                if st.button("🚀 Run Lime Local Explainability"):
                    with st.spinner(f"Running Local Explainability..."):
                        if data_selector == "Train":
                            results = lime_expl.explain_local(indices=[row_idx])
                        else:
                            results = lime_expl.explain_local(indices=[row_idx],dataset='test')
                        st.success("Local Explainability generated.")

                    res = results[0]
                    # print(res)
                    # st.image(res["png"], caption=f" Lime Local Plot (Row {row_idx})")
                    with open(res["html"], "r", encoding="utf-8") as f:
                        html = f.read()
                        html = html.replace("background-color: black", "background-color: white")
                        html = html.replace("color: black", "color: white")
                    override = """
                    <style>
                    body { background-color: white !important; color: black !important; }
                    div, span, p { color: black !important; }
                    svg text { fill: black !important; }
                    </style>
                    """
                    components.html(override +html, height=600)

                    explanation = build_lime_explanation_text(res)
                    st.markdown(explanation)

        with tab2:
            # ---- Dynamic import of explainers ----
            try:
                model_folder = f"Model_{self.model_version}"
                shap_module_name = "shap_explain"
                full_module_path = f"{model_folder}.{shap_module_name}"

                shap_expl = import_module(full_module_path)
                st.success(f"Loaded {full_module_path}")
            except Exception as e:
                st.error(f"❌ Could not import {shap_module_name}.py for model {self.model_version}")
                st.info(f"Tried path: {full_module_path.replace('.', '/')}.py")
                st.exception(e)
                return

            with st.expander("Global Explainability", expanded=False):
                st.write("""

                    """)

                with st.spinner(f"Running global explainability..."):
                    results = shap_expl.explain_global()
                st.success("Global explainability generated.")
                for output in results.get("outputs", []):
                    if output.endswith(".png"):
                        st.image(output, caption=f"SHAP Global Plot")
                    elif output.endswith(".json"):
                        st.markdown("**Top global features:**")
                        with open(output, "r") as f:
                            st.json(dict(list(json.load(f).items())[:10]))

            with st.expander("Local Explainability", expanded=False):
                st.write("""

                    """)

                train_data_path = self.data_dir / f"LLM_data_train_preprocessed_{self.model_version}.csv"
                test_data_path = self.data_dir / f"LLM_data_test_preprocessed_{self.model_version}.csv"

                train_data = pd.read_csv(train_data_path)
                test_data = pd.read_csv(test_data_path)

                data_selector2 = st.radio(
                    "Select a Data Source",
                    [
                        "Train",
                        "Test",
                    ],
                    index=1,
                    horizontal=True,
                )

                # overall_data = pd.concat([train_data, test_data])
                # overall_data.reset_index(inplace=True)
                # st.dataframe(overall_data.head(10))

                if data_selector2 == "Train":
                    overall_data = train_data
                else:
                    overall_data = test_data

                row_idx = st.number_input("Select a row index to explain", min_value=0, max_value=overall_data.shape[0],
                                          step=1, value=0, key='SHAP_local_select')

                with st.expander("View Selected Index", expanded=False):
                    st.dataframe(overall_data.iloc[row_idx])
                if st.button("🚀 Run SHAP Local Explainability"):
                    with st.spinner(f"Running Local Explainability..."):
                        if data_selector == "Train":
                            results = shap_expl.explain_local(indices=[row_idx],top_n=7)
                        else:
                            results = shap_expl.explain_local(indices=[row_idx],top_n=7, dataset='test')
                        st.success("Local Explainability generated.")


                    res = results[0]
                    st.image(res["plot"], caption=f" SHAP Local Plot (Row {row_idx})")
                    explanation = build_shap_explanation_text(res)
                    st.markdown(explanation)

    def show_cluster_analysis_page(self):
        st.header("Cluster Analysis")
        st.caption("")

        st.markdown("""
                    <style>
                    /* Center and evenly space Streamlit tabs */
                    div[data-baseweb="tab-list"] {
                        justify-content: space-evenly;
                    }
                    </style>
                """, unsafe_allow_html=True)

        selected = st.selectbox(
            "Choose which Clustering version to explore:",
            options=st.session_state.available_clustering_models,
            index=st.session_state.available_clustering_models.index(st.session_state.clustering_version)
            if st.session_state.clustering_version in st.session_state.available_clustering_models else 0,
            key="clustering_version_select"
        )

        # Apply selection immediately if changed
        if selected != self.clustering_version:
            self.set_clustering_version(selected)
            st.success(f"Clustering version set to **{self.clustering_version}**.")
            st.rerun()

        metadata_file = CLUSTER_RESULTS_ROOT / self.clustering_version / "info.json"
        with open(metadata_file, "r") as f:
            info = json.load(f)

        plot_path = CLUSTER_RESULTS_ROOT / self.clustering_version / "plots"

        with st.expander("Metadata Parameters", expanded=False):
            for section_name, section_data in info.items():
                st.markdown(f"### {section_name}")

                rows = []
                for key, val in section_data.items():
                    rows.append({"Parameter": key, "Value": format_value(val)})

                df_section = pd.DataFrame(rows)
                st.table(df_section)

        with st.expander("Principal Components", expanded=False):

            pc_png_files = []
            for p in sorted(plot_path.iterdir()):
                if p.suffix == ".png" and p.name.upper().startswith("PC"):
                    pc_png_files.append(p)

            for file in pc_png_files:
                st.markdown(f"### {file.stem}")

                st.image(str(file),width=800,caption=file.name)

                st.markdown("")

        with st.expander("K - Selection Plots", expanded=False):

            plot_files = [
                "consensus_cdf.png",
                "consensus_delta_cdf.png",
                "consensus_auc_cdf.png",
                "consensus_stability_curve.png",
                "consensus_PAC_curve.png",
            ]

            images = []
            titles = []

            for fname in plot_files:
                fpath = plot_path / fname
                if fpath.exists():
                    img = Image.open(fpath)
                    img = img.resize((900, 500))
                    images.append(img)
                    titles.append(fname.replace(".png", ""))
                else:
                    st.warning(f"Missing plot: {fname}")

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()

            for ax in axes:
                ax.axis("off")

            # # Add borders for aesthetic consistency
            # for ax in axes:
            #     for spine in ax.spines.values():
            #         spine.set_visible(True)
            #         spine.set_linewidth(1.2)
            #         spine.set_color("#666")

            for idx, img in enumerate(images):
                axes[idx].imshow(img)
                axes[idx].set_title(
                    titles[idx],
                    fontsize=14,
                    fontweight="bold",
                    pad=10
                )
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])

            best_k_file =  CLUSTER_RESULTS_ROOT / self.clustering_version / "best_k_selection.json"
            with open(best_k_file, "r") as f:
                best_k_info = json.load(f)

            summary_text = (
                f"Best-K Summary\n\n"
                f"- PAC Minimum: {best_k_info['pac_min']}\n"
                f"- PAC Knee:  {best_k_info['pac_kneedle']}\n"
                f"- Max Stability:  {best_k_info['stab_max']}\n"
                f"- Stability Knee:  {best_k_info['stab_kneedle']}\n"
                f"- AUC Knee:  {best_k_info['auc_kneedle']}\n"
                f"- AUC Knee:  {best_k_info['delta_auc_kneedle']}\n\n"
            )

            axes[-1].text(
                0.3, 0.7,
                summary_text,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
                wrap=True
            )
            axes[-1].set_xticks([])
            axes[-1].set_yticks([])

            plt.tight_layout()
            st.pyplot(fig)


        with st.expander("Significance Test Results", expanded=False):
            significance_plot_file = CLUSTER_RESULTS_ROOT/ self.clustering_version / "significance_tests" / "significance.png"
            st.image(str(significance_plot_file),width=800,caption=significance_plot_file.name)

            dunn_dir = CLUSTER_RESULTS_ROOT / self.clustering_version / "significance_tests"
            dunn_files = sorted([p for p in dunn_dir.iterdir() if p.name.startswith("dunn_") and p.suffix == ".png"])

            if len(dunn_files) == 0:
                st.warning("No Dunn matrix PNGs found.")
            else:
                n = len(dunn_files)
                cols = 2
                rows = math.ceil(n / cols)

                fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))

                if rows == 1:
                    axes = np.array([axes])
                axes = axes.flatten()

                for ax, file in zip(axes, dunn_files):
                    img = Image.open(file)
                    ax.imshow(img)
                    ax.axis("off")

                for i in range(len(dunn_files), len(axes)):
                    axes[i].axis("off")

                st.pyplot(fig)


        with st.expander("Cluster Profiles", expanded=False):

            profile_dir = CLUSTER_RESULTS_ROOT / self.clustering_version / "cluster_profiles"
            profile_files = sorted([p for p in profile_dir.iterdir() if p.name.startswith("profile_") and p.suffix == ".png"])

            if len(profile_files) == 0:
                st.warning("No profiles found")
            else:
                n = len(profile_files)
                cols = 2
                rows = math.ceil(n / cols)

                fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))

                if rows == 1:
                    axes = np.array([axes])
                axes = axes.flatten()

                for ax, file in zip(axes, profile_files):
                    img = Image.open(file)
                    ax.imshow(img)
                    ax.axis("off")

                for i in range(len(profile_files), len(axes)):
                    axes[i].axis("off")

                st.pyplot(fig)

        with st.expander("TD Contamination", expanded=False):
            contamination_file = CLUSTER_RESULTS_ROOT / self.clustering_version / "TD_Contamination_Report.png"
            st.image(str(contamination_file), width=800, caption=file.name)




def format_value(v):
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return v

def main():
    app = DemoApp()

    st.sidebar.title("Navigation")

    pages = {
        "🏠 Home": "home",
        "📈 Data & Model" : "overview",
        "📊 EDA": "eda",
        "🧮 Model Results": "results",
        "🔍 xAI - Explainability": "explainability",
        "🧩 Cluster Analysis": "clustering",
    }

    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_key

    # st.sidebar.markdown("---")
    # st.sidebar.markdown("###### Note: These fields update when new page is selected.")
    #
    # st.sidebar.markdown("### Data Version (current)")
    # st.sidebar.info(st.session_state.data_version)
    #
    # st.sidebar.markdown("### Model Version (current)")
    # st.sidebar.info(st.session_state.model_version)
    #
    # st.sidebar.markdown("### Model Info")
    # st.sidebar.caption(get_model_info(st.session_state.model_version))

    # Ensure artifacts exist for selected version
    if not app.load_results():
        st.error(f"Could not find results for **{st.session_state.model_version}** at `Results/{st.session_state.model_version}`.")
        # st.stop()

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
