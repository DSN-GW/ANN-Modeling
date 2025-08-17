import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from xgboost_model import XGBoostClassifier
from data_preprocessor import preprocess_prediction_data
from explainability_analysis import run_explainability_analysis
from visualization import create_visualizations

class ModelPredictor:
    def __init__(self):
        self.classifier = XGBoostClassifier()
        # Get the project root directory (two levels up from current file)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        self.results_dir = project_root / "Results" / "V1"
        self.predictions_dir = self.results_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        self.classifier.load_model()
        print("Model loaded successfully.")
    
    def preprocess_new_data(self, df, is_test_data=False):
        """
        Preprocess new data using saved preprocessing artifacts.
        If is_test_data=True, will try to load preprocessed test data first.
        """
        try:
            final_df = preprocess_prediction_data(df, is_test_data=is_test_data)
            return final_df
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run the training pipeline first to create preprocessing artifacts.")
            raise
    
    def predict_batch(self, df, is_test_data=False):
        processed_df = self.preprocess_new_data(df, is_test_data=is_test_data)
        
        X, _ = self.classifier.prepare_features(processed_df)
        
        predictions, probabilities = self.classifier.predict(X)
        
        results_df = processed_df.copy()
        results_df['predicted_td_or_asd'] = predictions
        results_df['prediction_probability_class_0'] = probabilities[:, 0]
        results_df['prediction_probability_class_1'] = probabilities[:, 1]
        results_df['prediction_confidence'] = np.max(probabilities, axis=1)
        
        return results_df
    
    def predict_single_text(self, text, subject_id="unknown"):
        single_df = pd.DataFrame({
            'sub': [subject_id],
            'profile': ['unknown'],
            'subject': [subject_id],
            'td_or_asd': [0],
            'SRS.Raw': [0],
            'FSR': [0],
            'BIS': [0],
            'avg_PE': [0],
            'free_response': [text],
            'LPA_Profile_grand_mean': [0],
            'LPA_Profile_ASD_only': [0]
        })
        
        result = self.predict_batch(single_df)
        return result.iloc[0]
    
    def create_prediction_visualization(self, results_df, save_path=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        prediction_counts = results_df['predicted_td_or_asd'].value_counts()
        ax1.pie(prediction_counts.values, labels=['TD', 'ASD'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Prediction Distribution')
        
        ax2.hist(results_df['prediction_confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Confidence Distribution')
        
        confidence_by_class = results_df.groupby('predicted_td_or_asd')['prediction_confidence'].mean()
        ax3.bar(['TD', 'ASD'], confidence_by_class.values, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax3.set_ylabel('Average Confidence')
        ax3.set_title('Average Confidence by Predicted Class')
        for i, v in enumerate(confidence_by_class.values):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        prob_diff = np.abs(results_df['prediction_probability_class_1'] - results_df['prediction_probability_class_0'])
        ax4.scatter(results_df.index, prob_diff, alpha=0.6, 
                   c=results_df['predicted_td_or_asd'], cmap='coolwarm')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Probability Difference')
        ax4.set_title('Prediction Certainty by Sample')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Certainty Threshold')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualization saved to: {save_path}")
        
        plt.close()
    
    def create_feature_contribution_plot(self, sample_result, top_n=15):
        feature_cols = [col for col in sample_result.index 
                       if col not in ['sub', 'profile', 'subject', 'td_or_asd', 'free_response',
                                     'predicted_td_or_asd', 'prediction_probability_class_0',
                                     'prediction_probability_class_1', 'prediction_confidence']]
        
        feature_values = sample_result[feature_cols]
        feature_importance = self.classifier.feature_importance
        
        contributions = []
        for feature in feature_cols:
            if feature in feature_importance:
                contribution = feature_values[feature] * feature_importance[feature]
                contributions.append((feature, contribution, feature_values[feature]))
        
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_contributions = contributions[:top_n]
        
        features, contribs, values = zip(*top_contributions)
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if c < 0 else 'blue' for c in contribs]
        bars = plt.barh(range(len(features)), contribs, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Contribution')
        plt.title(f'Top {top_n} Feature Contributions for Sample Prediction')
        plt.gca().invert_yaxis()
        
        for i, (bar, contrib, value) in enumerate(zip(bars, contribs, values)):
            plt.text(bar.get_width() + (0.001 if contrib >= 0 else -0.001), 
                    bar.get_y() + bar.get_height()/2,
                    f'{contrib:.3f} (val: {value:.2f})', 
                    va='center', ha='left' if contrib >= 0 else 'right', fontsize=8)
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.predictions_dir / 'sample_feature_contributions_v1.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_prediction_report(self, results_df):
        report = {
            'total_samples': len(results_df),
            'td_predictions': int((results_df['predicted_td_or_asd'] == 0).sum()),
            'asd_predictions': int((results_df['predicted_td_or_asd'] == 1).sum()),
            'average_confidence': float(results_df['prediction_confidence'].mean()),
            'high_confidence_predictions': int((results_df['prediction_confidence'] > 0.8).sum()),
            'low_confidence_predictions': int((results_df['prediction_confidence'] < 0.6).sum()),
            'prediction_summary': {
                'td_percentage': float((results_df['predicted_td_or_asd'] == 0).mean() * 100),
                'asd_percentage': float((results_df['predicted_td_or_asd'] == 1).mean() * 100)
            }
        }
        
        return report
    
    def save_predictions(self, results_df, filename='predictions_v1.csv'):
        output_path = self.predictions_dir / filename
        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        return output_path

def predict_from_file(data_path):
    print("="*60)
    print("STARTING MODEL V1 TEST DATA ANALYSIS PIPELINE")
    print("="*60)
    
    predictor = ModelPredictor()
    
    print("\nLoading trained model...")
    predictor.load_model()
    
    print(f"\nLoading test data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Test data loaded. Shape: {df.shape}")
    print(f"Test target distribution: {df['td_or_asd'].value_counts().to_dict()}")
    
    print("\nMaking predictions on test data...")
    # Use optimized preprocessing for test data (will load preprocessed data if available)
    results_df = predictor.predict_batch(df, is_test_data=True)
    
    # Extract true labels and predictions for evaluation
    y_true = results_df['td_or_asd']
    y_pred = results_df['predicted_td_or_asd']
    y_pred_proba = results_df['prediction_probability_class_1']
    
    print("\nEvaluating model performance on test data...")
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate ROC AUC if we have both classes
    roc_auc = None
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    test_metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'roc_auc': roc_auc
    }
    
    print("\nRunning explainability analysis on test data...")
    analyzer, explainability_report, text_analysis = run_explainability_analysis()
    
    print("\nGenerating comprehensive visualizations on test data...")
    viz_dir = create_visualizations()
    
    print("\nGenerating prediction-specific visualizations...")
    viz_path = predictor.predictions_dir / 'test_prediction_analysis_v1.png'
    predictor.create_prediction_visualization(results_df, viz_path)
    
    if len(results_df) > 0:
        print("\nCreating feature contribution plot for highest confidence prediction...")
        sample_idx = results_df['prediction_confidence'].idxmax()
        sample_result = results_df.loc[sample_idx]
        contrib_path = predictor.create_feature_contribution_plot(sample_result)
    
    print("\nSaving comprehensive results...")
    pred_path = predictor.save_predictions(results_df, 'test_predictions_v1.csv')
    
    # Save comprehensive test results for app.py
    comprehensive_results = {
        'test_metrics': test_metrics,
        'prediction_summary': predictor.generate_prediction_report(results_df),
        'explainability_summary': {
            'top_characteristics': sorted(
                explainability_report['characteristic_summary'].items(), 
                key=lambda x: x[1]['importance_score'], 
                reverse=True
            )[:10]
        }
    }
    
    results_path = predictor.results_dir / 'comprehensive_test_results_v1.json'
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("TEST DATA ANALYSIS PIPELINE COMPLETED")
    print("="*60)
    
    print(f"\nTEST PERFORMANCE METRICS:")
    print(f"- Test Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"- Test ROC AUC: {roc_auc:.4f}")
    print(f"- Confusion Matrix:")
    print(f"  [[{conf_matrix[0,0]}, {conf_matrix[0,1]}],")
    print(f"   [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
    
    print(f"\nTEST PREDICTION SUMMARY:")
    report = comprehensive_results['prediction_summary']
    print(f"- Total test samples: {report['total_samples']}")
    print(f"- TD predictions: {report['td_predictions']} ({report['prediction_summary']['td_percentage']:.1f}%)")
    print(f"- ASD predictions: {report['asd_predictions']} ({report['prediction_summary']['asd_percentage']:.1f}%)")
    print(f"- Average confidence: {report['average_confidence']:.3f}")
    
    print(f"\nTOP 3 MOST IMPORTANT CHARACTERISTICS:")
    top_chars = comprehensive_results['explainability_summary']['top_characteristics'][:3]
    for i, (char, data) in enumerate(top_chars, 1):
        print(f"{i}. {char}: {data['importance_score']:.4f}")
    
    print(f"\nResults saved to: {predictor.results_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"Comprehensive results saved to: {results_path}")
    
    return results_df, comprehensive_results

def predict_single_sample(text, subject_id="sample"):
    predictor = ModelPredictor()
    predictor.load_model()
    
    result = predictor.predict_single_text(text, subject_id)
    
    print(f"\nPrediction for '{subject_id}':")
    print(f"Text: {text}")
    print(f"Predicted class: {'ASD' if result['predicted_td_or_asd'] == 1 else 'TD'}")
    print(f"Confidence: {result['prediction_confidence']:.3f}")
    print(f"Probabilities - TD: {result['prediction_probability_class_0']:.3f}, ASD: {result['prediction_probability_class_1']:.3f}")
    
    return result

if __name__ == "__main__":
    # Get the project root directory (two levels up from current file)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / "Data_v1" / "LLM_data_test.csv"
    
    if data_path.exists():
        results_df, report = predict_from_file(str(data_path))
    else:
        print(f"Test data file not found: {data_path}")
        print("Please run train.py first to create the test data split.")
        
        sample_text = "He likes sports and healthy food. He enjoys playing guitar but doesn't like art."
        result = predict_single_sample(sample_text, "demo_sample")