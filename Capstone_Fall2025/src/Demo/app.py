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
import plotly.graph_objects as go

# Get the current file's directory and ensure it's absolute
current_dir = Path(__file__).resolve().parent
model_v2_dir = current_dir.parent / "Model_V1"

# Add the Model_V1 directory to the path
if str(model_v2_dir) not in sys.path:
    sys.path.insert(0, str(model_v2_dir))

# Try to import with better error handling
try:
    from predict import ModelPredictor, predict_single_sample
except ImportError as e:
    # Try alternative import method
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("predict", model_v2_dir / "predict.py")
        predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_module)
        ModelPredictor = predict_module.ModelPredictor
        predict_single_sample = predict_module.predict_single_sample
    except Exception as e2:
        st.error(f"Failed to import predict module: {e}")
        st.error(f"Alternative import also failed: {e2}")
        st.error(f"Current directory: {current_dir}")
        st.error(f"Model_V1 directory path: {model_v2_dir}")
        st.error(f"Model_V1 exists: {model_v2_dir.exists()}")
        st.error(f"Current sys.path: {sys.path[:3]}")
        st.error(f"Files in Model_V1: {list(model_v2_dir.glob('*.py'))}")
        st.stop()

st.set_page_config(
    page_title="TD/ASD Classification Model V1 - Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DemoApp:
    def __init__(self):
        # Get the absolute path to the results directory
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        self.results_dir = project_root / "Results" / "V1"
        self.viz_dir = self.results_dir / "visualizations"
        self.predictor = None
        
    def load_results(self):
        try:
            explainability_path = self.results_dir / 'explainability_analysis_v2.json'
            training_results_path = self.results_dir / 'training_results_v2.json'
            test_results_path = self.results_dir / 'comprehensive_test_results_v2.json'
            
            with open(explainability_path, 'r') as f:
                self.explainability_data = json.load(f)
            
            with open(training_results_path, 'r') as f:
                self.training_results = json.load(f)
            
            # Try to load test results if available
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
        if self.predictor is None:
            try:
                self.predictor = ModelPredictor()
                self.predictor.load_model()
                return True
            except:
                return False
        return True
    
    def show_home_page(self):
        st.title("🧠 TD/ASD Classification Model V1")
        st.markdown("### Advanced Text Analysis for Autism Spectrum Disorder Classification")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Use test accuracy if available, otherwise use CV mean
            if self.test_results and 'test_metrics' in self.test_results:
                accuracy = self.test_results['test_metrics']['accuracy']
                st.metric("Test Accuracy", f"{accuracy:.3f}")
            else:
                cv_mean = self.training_results.get('cv_mean', 0)
                st.metric("CV Mean Accuracy", f"{cv_mean:.3f}")
        
        with col2:
            cv_mean = self.training_results.get('cv_mean', 0)
            st.metric("Cross-Validation Score", f"{cv_mean:.3f}")
        
        with col3:
            # Use test ROC AUC if available
            if self.test_results and 'test_metrics' in self.test_results:
                roc_auc = self.test_results['test_metrics'].get('roc_auc', 0)
                if roc_auc:
                    st.metric("Test ROC AUC", f"{roc_auc:.3f}")
                else:
                    st.metric("CV Std", f"{self.training_results.get('cv_std', 0):.3f}")
            else:
                st.metric("CV Std", f"{self.training_results.get('cv_std', 0):.3f}")
        
        st.markdown("---")
        
        st.markdown("""
        ## Model Overview
        
        This advanced machine learning model uses **XGBoost** to classify individuals as Typically Developing (TD) or having Autism Spectrum Disorder (ASD) based on free-response text analysis.
        
        ### Key Features:
        - **Characteristic-based Feature Extraction**: Uses Claude 3.5 Sonnet to extract features related to 11 specific characteristics
        - **NLP Text Analysis**: Comprehensive text preprocessing including sentiment, cohesiveness, and linguistic features
        - **Explainable AI**: SHAP-based feature importance and contribution analysis
        - **Interactive Predictions**: Real-time text analysis and classification
        
        ### Model Pipeline:
        1. **Text Preprocessing**: Extract characteristic-based and NLP features
        2. **Feature Engineering**: Create comprehensive feature set from free response text
        3. **XGBoost Classification**: Train robust gradient boosting model
        4. **Explainability Analysis**: Generate interpretable insights
        5. **Visualization**: Create comprehensive analysis charts
        """)
        
        if st.button("🚀 Explore Model Results"):
            st.session_state.page = "results"
            st.rerun()
    
    def show_results_page(self):
        st.title("📊 Model Results & Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Performance", "Feature Importance", "Characteristic Analysis", "Visualizations", "Prediction Analysis"])
        
        with tab1:
            st.subheader("Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Use test classification report if available
                if self.test_results and 'test_metrics' in self.test_results:
                    st.json(self.test_results['test_metrics']['classification_report'])
                else:
                    st.info("Classification report not available. Run predict.py to generate test metrics.")
            
            with col2:
                # Use test confusion matrix if available
                if self.test_results and 'test_metrics' in self.test_results:
                    cm = np.array(self.test_results['test_metrics']['confusion_matrix'])
                elif 'confusion_matrix' in self.training_results:
                    cm = np.array(self.training_results['confusion_matrix'])
                else:
                    st.info("Confusion matrix not available. Run predict.py to generate test metrics.")
                    return
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['TD', 'ASD'], yticklabels=['TD', 'ASD'], ax=ax)
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
        
        with tab2:
            st.subheader("Top Feature Importance")
            
            feature_importance = self.training_results['feature_importance']
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
            
            fig = px.bar(features_df, x='Importance', y='Feature', orientation='h',
                        title='Top 20 Most Important Features')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Characteristic Analysis")
            
            char_summary = self.explainability_data['characteristic_summary']
            
            char_df = pd.DataFrame([
                {
                    'Characteristic': char.replace('_', ' ').title(),
                    'Importance Score': data['importance_score'],
                    'Feature Count': data['feature_count'],
                    'Rank': data['rank']
                }
                for char, data in char_summary.items()
            ]).sort_values('Rank')
            
            st.dataframe(char_df, use_container_width=True)
            
            fig = px.bar(char_df, x='Characteristic', y='Importance Score',
                        title='Characteristic Importance Ranking')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Model Visualizations")
            
            viz_files = [
                ('feature_importance_v2.png', 'Feature Importance'),
                ('characteristic_importance_v2.png', 'Characteristic Importance'),
                ('td_vs_asd_comparison_v2.png', 'TD vs ASD Comparison'),
                ('model_performance_v2.png', 'Model Performance'),
                ('confusion_matrix_v2.png', 'Confusion Matrix'),
                ('characteristic_ranking_v2.png', 'Characteristic Ranking')
            ]
            
            for filename, title in viz_files:
                viz_path = self.viz_dir / filename
                if viz_path.exists():
                    st.subheader(title)
                    image = Image.open(viz_path)
                    st.image(image, use_container_width=True)
                else:
                    st.warning(f"Visualization not found: {filename}")
        
        with tab5:
            st.subheader("Prediction Analysis Visualizations")
            
            # Path to prediction visualizations
            predictions_dir = self.results_dir / 'predictions'
            
            # Test prediction analysis visualization
            test_pred_path = predictions_dir / 'test_prediction_analysis_v2.png'
            if test_pred_path.exists():
                st.subheader("Test Data Prediction Analysis")
                st.markdown("""
                This visualization shows the comprehensive analysis of predictions on the test dataset, 
                including prediction distributions, confidence scores, and performance metrics.
                """)
                image = Image.open(test_pred_path)
                st.image(image, use_container_width=True)
            else:
                st.warning("Test prediction analysis visualization not found. Run predict.py to generate.")
            
            # Sample feature contributions visualization
            sample_contrib_path = predictions_dir / 'sample_feature_contributions_v2.png'
            if sample_contrib_path.exists():
                st.subheader("Sample Feature Contributions")
                st.markdown("""
                This visualization shows the feature contribution analysis for the highest confidence prediction, 
                demonstrating how individual features influence the model's decision.
                """)
                image = Image.open(sample_contrib_path)
                st.image(image, use_container_width=True)
            else:
                st.warning("Sample feature contributions visualization not found. Run predict.py to generate.")
            
            # Show prediction summary if available
            if self.test_results and 'prediction_summary' in self.test_results:
                st.subheader("Test Prediction Summary")
                summary = self.test_results['prediction_summary']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", summary['total_samples'])
                with col2:
                    st.metric("TD Predictions", summary['td_predictions'])
                with col3:
                    st.metric("ASD Predictions", summary['asd_predictions'])
                with col4:
                    st.metric("Avg Confidence", f"{summary['average_confidence']:.3f}")
                
                # Show prediction distribution
                pred_data = pd.DataFrame({
                    'Class': ['TD', 'ASD'],
                    'Count': [summary['td_predictions'], summary['asd_predictions']],
                    'Percentage': [
                        summary['prediction_summary']['td_percentage'],
                        summary['prediction_summary']['asd_percentage']
                    ]
                })
                
                fig = px.pie(pred_data, values='Count', names='Class', 
                           title='Test Prediction Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Display test predictions CSV data
            test_predictions_path = predictions_dir / 'test_predictions_v2.csv'
            if test_predictions_path.exists():
                st.subheader("Test Predictions Data")
                st.markdown("""
                Below is a sample of the test predictions with confidence scores and probabilities.
                """)
                
                # Load and display the predictions data
                try:
                    predictions_df = pd.read_csv(test_predictions_path)
                    
                    # Show summary statistics
                    st.markdown("**Prediction Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions_df))
                    with col2:
                        avg_conf = predictions_df['prediction_confidence'].mean()
                        st.metric("Average Confidence", f"{avg_conf:.3f}")
                    with col3:
                        high_conf = (predictions_df['prediction_confidence'] > 0.9).sum()
                        st.metric("High Confidence (>0.9)", high_conf)
                    
                    # Show sample of predictions
                    st.markdown("**Sample Predictions (first 10 rows):**")
                    display_cols = ['sub', 'free_response', 'predicted_td_or_asd', 
                                  'prediction_confidence', 'prediction_probability_class_0', 
                                  'prediction_probability_class_1']
                    available_cols = [col for col in display_cols if col in predictions_df.columns]
                    
                    if available_cols:
                        sample_df = predictions_df[available_cols].head(10)
                        # Format the predicted_td_or_asd column
                        if 'predicted_td_or_asd' in sample_df.columns:
                            sample_df['predicted_td_or_asd'] = sample_df['predicted_td_or_asd'].map({0: 'TD', 1: 'ASD'})
                        st.dataframe(sample_df, use_container_width=True)
                    
                    # Download button for the full predictions file
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Test Predictions",
                        data=csv,
                        file_name="test_predictions_v2.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error loading predictions data: {str(e)}")
            else:
                st.warning("Test predictions CSV file not found. Run predict.py to generate.")
    
    def show_prediction_page(self):
        st.title("🔮 Make Predictions")
        
        tab1, tab2 = st.tabs(["Single Text Prediction", "Batch Prediction"])
        
        with tab1:
            st.subheader("Analyze Single Text Sample")
            
            text_input = st.text_area(
                "Enter free response text:",
                placeholder="Example: He likes sports and healthy food. He enjoys playing guitar but doesn't like art.",
                height=150
            )
            
            subject_id = st.text_input("Subject ID (optional):", value="demo_sample")
            
            if st.button("🔍 Analyze Text"):
                if text_input.strip():
                    if self.initialize_predictor():
                        with st.spinner("Analyzing text..."):
                            try:
                                result = self.predictor.predict_single_text(text_input, subject_id)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    prediction = "ASD" if result['predicted_td_or_asd'] == 1 else "TD"
                                    st.metric("Prediction", prediction)
                                
                                with col2:
                                    confidence = result['prediction_confidence']
                                    st.metric("Confidence", f"{confidence:.3f}")
                                
                                with col3:
                                    prob_asd = result['prediction_probability_class_1']
                                    st.metric("ASD Probability", f"{prob_asd:.3f}")
                                
                                st.markdown("---")
                                
                                st.subheader("Probability Breakdown")
                                prob_data = pd.DataFrame({
                                    'Class': ['TD', 'ASD'],
                                    'Probability': [
                                        result['prediction_probability_class_0'],
                                        result['prediction_probability_class_1']
                                    ]
                                })
                                
                                fig = px.bar(prob_data, x='Class', y='Probability',
                                           title='Classification Probabilities')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.subheader("Feature Analysis")
                                feature_cols = [col for col in result.index 
                                              if col not in ['sub', 'profile', 'subject', 'td_or_asd', 'free_response',
                                                           'predicted_td_or_asd', 'prediction_probability_class_0',
                                                           'prediction_probability_class_1', 'prediction_confidence']]
                                
                                non_zero_features = []
                                for col in feature_cols:
                                    if result[col] != 0:
                                        non_zero_features.append({'Feature': col, 'Value': result[col]})
                                
                                if non_zero_features:
                                    features_df = pd.DataFrame(non_zero_features)
                                    st.dataframe(features_df, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                    else:
                        st.error("Failed to load model. Please ensure the model is trained.")
                else:
                    st.warning("Please enter some text to analyze.")
        
        with tab2:
            st.subheader("Batch Prediction from File")
            
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Data Preview:")
                    st.dataframe(df.head())
                    
                    if st.button("🚀 Run Batch Prediction"):
                        if self.initialize_predictor():
                            with st.spinner("Processing batch predictions..."):
                                try:
                                    results_df = self.predictor.predict_batch(df)
                                    
                                    st.success("Batch prediction completed!")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        td_count = (results_df['predicted_td_or_asd'] == 0).sum()
                                        st.metric("TD Predictions", td_count)
                                    
                                    with col2:
                                        asd_count = (results_df['predicted_td_or_asd'] == 1).sum()
                                        st.metric("ASD Predictions", asd_count)
                                    
                                    with col3:
                                        avg_confidence = results_df['prediction_confidence'].mean()
                                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                                    
                                    st.subheader("Prediction Results")
                                    st.dataframe(results_df[['sub', 'free_response', 'predicted_td_or_asd', 
                                                           'prediction_confidence']].head(20))
                                    
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results",
                                        data=csv,
                                        file_name="batch_predictions_v2.csv",
                                        mime="text/csv"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error during batch prediction: {str(e)}")
                        else:
                            st.error("Failed to load model.")
                            
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    def show_explainability_page(self):
        st.title("🔍 Model Explainability")
        
        tab1, tab2, tab3 = st.tabs(["Characteristic Contributions", "TD vs ASD Patterns", "Feature Analysis"])
        
        with tab1:
            st.subheader("How Each Characteristic Contributes to Classification")
            
            char_analysis = self.explainability_data['characteristic_analysis']
            
            contrib_data = []
            for char, data in char_analysis.items():
                contrib_data.append({
                    'Characteristic': char.replace('_', ' ').title(),
                    'Total Importance': data['total_importance'],
                    'Feature Count': data['feature_count']
                })
            
            contrib_df = pd.DataFrame(contrib_data).sort_values('Total Importance', ascending=False)
            
            fig = px.scatter(contrib_df, x='Feature Count', y='Total Importance', 
                           size='Total Importance', hover_name='Characteristic',
                           title='Characteristic Importance vs Feature Count')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(contrib_df, use_container_width=True)
        
        with tab2:
            st.subheader("Differences Between TD and ASD Groups")
            
            td_patterns = self.explainability_data['class_patterns']['td_patterns']
            asd_patterns = self.explainability_data['class_patterns']['asd_patterns']
            
            comparison_data = []
            for char in td_patterns.keys():
                td_mentioned = td_patterns[char].get(f'{char}_mentioned', 0)
                asd_mentioned = asd_patterns[char].get(f'{char}_mentioned', 0)
                td_sentiment = td_patterns[char].get(f'{char}_sentiment', 0)
                asd_sentiment = asd_patterns[char].get(f'{char}_sentiment', 0)
                
                comparison_data.append({
                    'Characteristic': char.replace('_', ' ').title(),
                    'TD Mention Rate': td_mentioned,
                    'ASD Mention Rate': asd_mentioned,
                    'TD Sentiment': td_sentiment,
                    'ASD Sentiment': asd_sentiment
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig1 = px.bar(comparison_df, x='Characteristic', 
                         y=['TD Mention Rate', 'ASD Mention Rate'],
                         title='Mention Rates: TD vs ASD',
                         barmode='group')
            fig1.update_xaxes(tickangle=45)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.bar(comparison_df, x='Characteristic', 
                         y=['TD Sentiment', 'ASD Sentiment'],
                         title='Sentiment Scores: TD vs ASD',
                         barmode='group')
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("Detailed Feature Analysis")
            
            top_features = self.explainability_data['model_performance']['top_features']
            
            if top_features:
                features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                
                fig = px.treemap(features_df, path=['Feature'], values='Importance',
                               title='Feature Importance Treemap')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(features_df, use_container_width=True)

def main():
    app = DemoApp()
    
    st.sidebar.title("Navigation")
    
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    pages = {
        "🏠 Home": "home",
        "📊 Results": "results", 
        "🔮 Predictions": "predictions",
        "🔍 Explainability": "explainability"
    }
    
    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_key
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model V1 Info")
    st.sidebar.info("Advanced TD/ASD Classification using XGBoost and NLP features")
    
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