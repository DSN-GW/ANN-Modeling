# TD/ASD Classification Model V2

Advanced machine learning system for classifying Typically Developing (TD) vs Autism Spectrum Disorder (ASD) individuals based on free response text analysis using XGBoost and comprehensive NLP features.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DSN-GW
```

2. **Install dependencies (root):**
```bash
pip install -r requirements.txt
```

3. **Train the model (V2):**
```bash
cd ANN-Modeling\Code\Model_V2
python train.py
```

4. **Run predictions (V2):**
```bash
python predict.py
```

5. **Launch the web application:**
```bash
cd ..\Demo
streamlit run app.py
```

Note: The Streamlit app has its own dependencies in `ANN-Modeling\Code\Demo\requirements.txt`.

## 📁 Project Structure

```
DSN-GW/
├── ANN-Modeling/
│   ├── Code/
│   │   ├── Agent/                 # Optional Claude Sonnet agent utilities (.env-driven)
│   │   │   ├── sonnet_agent.py
│   │   │   ├── run_sonnet_agent.py
│   │   │   └── demo_multiple_agents.py
│   │   ├── Model_V2/              # Main model implementation (current)
│   │   │   ├── train.py           # Training pipeline
│   │   │   ├── predict.py         # Prediction pipeline (batch/file)
│   │   │   ├── visualization.py   # Visualization generation
│   │   │   ├── data_preprocessor.py
│   │   │   ├── xgboost_model.py
│   │   │   └── explainability_analysis.py
│   │   ├── Model_V1/              # Legacy model (kept for reference)
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   ├── visualization.py
│   │   │   ├── data_preprocessor.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── nlp_features.py
│   │   │   └── explainability_analysis.py
│   │   └── Demo/                  # Streamlit web application
│   │       ├── app.py             # Main web app
│   │       └── requirements.txt   # App dependencies
│   ├── Results/
│   │   ├── V1/                    # Legacy results (for Model V1)
│   │   └── V2/                    # Model V2 results and visualizations
│   │       ├── training_results_v2.json
│   │       ├── test_results_v2.json
│   │       ├── feature_importance_v2.json
│   │       ├── explainability_analysis_v2.json
│   │       ├── feature_names_v2.pkl
│   │       ├── scaler_v2.pkl
│   │       ├── xgboost_model_v2.pkl
│   │       ├── predictions/
│   │       │   └── test_predictions_v2.csv
│   │       └── visualizations/
│   │           ├── model_performance_v2.png
│   │           ├── confusion_matrix_v2.png
│   │           ├── feature_importance_v2.png
│   │           ├── feature_importance_by_target_v2.png
│   │           ├── characteristic_importance_v2.png
│   │           └── td_vs_asd_comparison_v2.png
│   ├── Paper/
│   │   └── 1-Socail_Paper/        # Research artifacts (PDFs and summaries)
│   ├── Scratch_Codo/              # Experimental code and artifacts (legacy)
│   │   ├── Model_V1/
│   │   └── V1/
│   └── data/
│       ├── Data_v1/               # Training and test CSVs (v1/v2 preprocessed outputs)
│       └── Data_v2/               # Trial-level dataset (optional)
├── requirements.txt               # Project-wide dependencies (root)
└── README.md                      # This file
```

## 🔧 Features

### Model Capabilities
- **Characteristic-based Feature Extraction**: Uses Claude 3.5 Sonnet (via AWS Bedrock) to derive 11 characteristic-driven features
- **NLP Text Analysis**: Sentiment, readability, lexical diversity, and structural features
- **XGBoost Classification**: Gradient boosting with tuned regularization
- **Explainable AI**: SHAP-based feature importance and contribution analysis
- **Interactive Predictions**: Web UI for input and batch predictions

### Web Application Features
- **Model Performance Dashboard**: Accuracy and confusion matrix
- **Feature Importance Analysis**: Interactive visualizations of model features
- **Characteristic Analysis**: Detailed breakdown of characteristic contributions
- **Prediction Interface**: Real-time text classification with confidence scores
- **Batch Prediction**: Upload CSV files for bulk analysis
- **Explainability Insights**: Understand model decisions and feature contributions

## 📊 Model Performance

See the latest artifacts in:
- `ANN-Modeling/Results/V2/test_results_v2.json`
- `ANN-Modeling/Results/V2/visualizations/model_performance_v2.png`
- `ANN-Modeling/Results/V2/visualizations/confusion_matrix_v2.png`

These are generated by the current V2 training and prediction pipelines and reflect the most up-to-date performance.

## 🛠️ Technical Details

### Dependencies
- **Core ML**: XGBoost, scikit-learn, SHAP
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: streamlit
- **NLP**: textblob, nltk, textstat
- **Cloud**: boto3 (for Claude 3.5 Sonnet via AWS Bedrock)

### Path Management
All critical paths use `pathlib.Path` for cross-platform compatibility.

### Target Leakage Prevention
- Fixed double train-test splits
- Enhanced feature selection
- Prevented feature extraction leakage
- Improved regularization

## 🚀 Usage

### Training the Model (V2)
```bash
cd ANN-Modeling\Code\Model_V2
python train.py
```

### Making Predictions (from preprocessed test data)
```bash
python predict.py
```
This will also run explainability analysis and generate visualizations and CSV predictions under `ANN-Modeling\Results\V2`.

### Programmatic Prediction (batch or single row)
```python
import pandas as pd
from predict import ModelPredictor

# Example: single text (provide required columns)
df = pd.DataFrame([
    {
        "FSR": 0.5,                 # numeric
        "avg_PE": 0.2,              # numeric
        "free_response": "Sample text here",
        # optional during prediction, but if provided enables accuracy metrics
        # "td_or_asd": 1             # 1 = ASD, 0 = TD
    }
])

predictor = ModelPredictor()
predictor.load_model()
results_df = predictor.predict_batch(df, is_test_data=False)
print(results_df[["predicted_td_or_asd", "prediction_probability"]])
```

### Running the Web App
```bash
cd ANN-Modeling\Code\Demo
streamlit run app.py
```

## 🤖 Claude Sonnet Agent (Optional)
- Location: `ANN-Modeling\Code\Agent`
- Used by the feature extractor to enrich characteristic-based features via AWS Bedrock.
- Configure environment in `ANN-Modeling\Code\Agent\.env` with:
  - `aws_access_key_id`
  - `aws_secret_access_key`
  - `aws_session_token` (if applicable)

## 📈 Results

Key outputs saved to `ANN-Modeling/Results/V2/` include:
- `xgboost_model_v2.pkl`, `scaler_v2.pkl`, `feature_names_v2.pkl`
- `training_results_v2.json`, `test_results_v2.json`
- `feature_importance_v2.json`, `explainability_analysis_v2.json`
- `predictions/test_predictions_v2.csv`
- Visuals in `visualizations/`: model performance, confusion matrix, feature importance (overall/by target), characteristic importance, TD vs ASD comparison

## 🔍 Model Explainability

The model provides comprehensive explainability through:
- **SHAP Analysis**: Feature importance and contribution analysis
- **Characteristic Breakdown**: How each characteristic contributes to classification
- **TD vs ASD Patterns**: Differences between typically developing and ASD groups
- **Feature Analysis**: Detailed examination of individual features

## 🛡️ Security and Best Practices

- **No Target Leakage**: Comprehensive analysis performed to prevent data leakage
- **Proper Validation**: Cross-validation and holdout test sets
- **Regularization**: XGBoost parameters tuned to prevent overfitting
- **Feature Selection**: Correlation-based filtering and importance-based selection

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

[Add contact information here]