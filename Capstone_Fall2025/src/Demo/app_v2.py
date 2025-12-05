import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px

# ------------------------------
# Setup paths and imports
# ------------------------------
current_dir = Path(__file__).resolve().parent
model_v2_dir = current_dir.parent / "Model_V2"

if str(model_v2_dir) not in sys.path:
    sys.path.insert(0, str(model_v2_dir))

try:
    from predict import ModelPredictor
except ImportError as e:
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict", model_v2_dir / "predict.py")
    predict_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_module)
    ModelPredictor = predict_module.ModelPredictor

st.set_page_config(
    page_title="TD/ASD Classification Model V2 - Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Demo App Class
# ------------------------------
class DemoApp:
    def __init__(self):
        project_root = current_dir.parent.parent
        self.results_dir = project_root / "Results" / "V2"
        self.viz_dir = self.results_dir / "visualizations"
        self.predictor = None

        # placeholders
        self.training_results = {}
        self.test_results = {}
        self.explainability_data = {}
        self.feature_importance = {}

    def load_results(self):
        try:
            explainability_path = self.results_dir / "explainability_analysis_v2.json"
            training_results_path = self.results_dir / "training_results_v2.json"
            test_results_path = self.results_dir / "test_results_v2.json"
            feature_importance_path = self.results_dir / "feature_importance_v2.json"

            if explainability_path.exists():
                with open(explainability_path, "r") as f:
                    self.explainability_data = json.load(f)

            if training_results_path.exists():
                with open(training_results_path, "r") as f:
                    self.training_results = json.load(f)

            if feature_importance_path.exists():
                with open(feature_importance_path, "r") as f:
                    self.feature_importance = json.load(f)

            if test_results_path.exists():
                with open(test_results_path, "r") as f:
                    self.test_results = json.load(f)

            return True
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            return False

    def initialize_predictor(self):
        if self.predictor is None:
            try:
                self.predictor = ModelPredictor()
                self.predictor.load_model()
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return True

    # ------------------------------
    # Pages
    # ------------------------------
    def show_home_page(self):
        st.title("🧠 TD/ASD Classification Model V2")
        st.markdown("### Advanced Text Analysis for Autism Spectrum Disorder Classification")

        col1, col2, col3 = st.columns(3)
        with col1:
            if self.test_results and "accuracy" in self.test_results:
                st.metric("Test Accuracy", f"{self.test_results['accuracy']:.3f}")
            else:
                cv_mean = self.training_results.get("cv_accuracy_mean", 0)
                st.metric("CV Mean Accuracy", f"{cv_mean:.3f}")

        with col2:
            cv_mean = self.training_results.get("cv_accuracy_mean", 0)
            st.metric("Cross-Validation Score", f"{cv_mean:.3f}")

        with col3:
            if self.training_results.get("cv_accuracy_std"):
                st.metric("CV Std", f"{self.training_results['cv_accuracy_std']:.3f}")
            else:
                st.metric("CV Std", "N/A")

        st.markdown("---")

        st.markdown("""
        ## Model Overview
        This advanced machine learning model uses **XGBoost** to classify individuals as TD or ASD based on free-response text.

        **Pipeline Steps**:
        1. Text Preprocessing  
        2. Feature Engineering  
        3. XGBoost Training  
        4. Explainability (SHAP + characteristic analysis)  
        5. Visualization  
        """)

        if st.button("🚀 Explore Model Results"):
            st.session_state.page = "results"
            st.rerun()

    def show_results_page(self):
        st.title("📊 Model Results & Analysis")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Model Performance", "Feature Importance", "Characteristic Analysis", "Visualizations", "Prediction Analysis"]
        )

        # -------- Tab 1: Model Performance --------
        with tab1:
            st.subheader("Model Performance Metrics")
            col1, col2 = st.columns(2)

            with col1:
                if self.test_results and "classification_report" in self.test_results:
                    st.json(self.test_results["classification_report"])
                else:
                    st.info("Classification report not available.")

            with col2:
                if self.test_results and "confusion_matrix" in self.test_results:
                    cm = np.array(self.test_results["confusion_matrix"])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["TD", "ASD"], yticklabels=["TD", "ASD"], ax=ax)
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                else:
                    st.info("Confusion matrix not available.")

        # -------- Tab 2: Feature Importance --------
        with tab2:
            st.subheader("Top Feature Importance")
            if self.feature_importance:
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
                features_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                fig = px.bar(features_df, x="Importance", y="Feature", orientation="h")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance data not available.")

        # -------- Tab 3: Characteristic Analysis --------
        with tab3:
            st.subheader("Characteristic Analysis")
            if "characteristic_summary" in self.explainability_data:
                char_summary = self.explainability_data["characteristic_summary"]
                char_df = pd.DataFrame([
                    {"Characteristic": c.replace("_", " ").title(),
                     "Importance Score": d["importance_score"],
                     "Feature Count": d["feature_count"],
                     "Rank": d["rank"]}
                    for c, d in char_summary.items()
                ]).sort_values("Rank")
                st.dataframe(char_df, use_container_width=True)
                fig = px.bar(char_df, x="Characteristic", y="Importance Score")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Characteristic analysis not available.")

        # -------- Tab 4: Visualizations --------
        with tab4:
            st.subheader("Model Visualizations")
            viz_files = [
                ("feature_importance_by_target_v2.png", "Feature Importance by Target"),
                ("td_vs_asd_comparison_v2.png", "TD vs ASD Comparison"),
                ("model_performance_v2.png", "Model Performance"),
                ("confusion_matrix_v2.png", "Confusion Matrix"),
            ]
            for filename, title in viz_files:
                viz_path = self.viz_dir / filename
                if viz_path.exists():
                    st.subheader(title)
                    image = Image.open(viz_path)
                    st.image(image, use_container_width=True)
                else:
                    st.warning(f"Visualization not found: {filename}")

        # -------- Tab 5: Prediction Analysis --------
        with tab5:
            st.subheader("Prediction Analysis Visualizations")
            predictions_dir = self.results_dir / "predictions"
            test_pred_path = predictions_dir / "test_prediction_analysis_v2.png"
            if test_pred_path.exists():
                st.image(Image.open(test_pred_path), caption="Test Data Prediction Analysis")
            test_predictions_path = predictions_dir / "test_predictions_v2.csv"
            if test_predictions_path.exists():
                predictions_df = pd.read_csv(test_predictions_path)
                st.dataframe(predictions_df.head(10))

    def show_prediction_page(self):
        st.title("🔮 Make Predictions")
        text_input = st.text_area("Enter free response text:")
        if st.button("Analyze Text"):
            if self.initialize_predictor() and text_input.strip():
                result = self.predictor.predict_single_text(text_input, "demo_sample")
                prediction = "ASD" if result["predicted_td_or_asd"] == 1 else "TD"
                st.metric("Prediction", prediction)

    def show_explainability_page(self):
        st.title("🔍 Model Explainability")
        if self.explainability_data:
            st.json(self.explainability_data)
        else:
            st.warning("Explainability data not available.")

# ------------------------------
# Main
# ------------------------------
def main():
    app = DemoApp()
    st.sidebar.title("Navigation")
    if "page" not in st.session_state:
        st.session_state.page = "home"
    pages = {"🏠 Home": "home", "📊 Results": "results",
             "🔮 Predictions": "predictions", "🔍 Explainability": "explainability"}
    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_key
    if not app.load_results():
        st.error("Could not load model results. Please ensure the model has been trained.")
        return
    if st.session_state.page == "home":
        app.show_home_page()
    elif st.session_state.page == "results":
        app.show_results_page()
    elif st.session_state.page == "predictions":
        app.show_prediction_page()
    elif st.session_state.page == "explainability":
        app.show_explainability_page()

if __name__ == "__main__":
    main()
