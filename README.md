# TD/ASD Classification Model V2

Advanced machine learning system for classifying Typically Developing (TD) vs Autism Spectrum Disorder (ASD) individuals based on free response text analysis using XGBoost and comprehensive NLP features.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DSN-GW
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train the model:**
```bash
cd ANN-Modeling/Code/Model_V1
python train.py
```

4. **Run predictions:**
```bash
python predict.py
```

5. **Launch the web application:**
```bash
cd ../Demo
streamlit run app.py
```

## 📁 Project Structure

```
DSN-GW/
├── ANN-Modeling/
│   ├── Code/
│   │   ├── Model_V2/           # Main model implementation
│   │   │   ├── train.py        # Training pipeline
│   │   │   ├── predict.py      # Prediction pipeline
│   │   │   ├── visualization.py # Visualization generation
│   │   │   └── requirements.txt # Model dependencies
│   │   └── Demo/               # Streamlit web application
│   │       ├── app.py          # Main web app
│   │       └── requirements.txt # App dependencies
│   ├── Results/
│   │   └── V2/                # Model results and visualizations
│   └── data/
│       └── Data_v1/           # Training and test data
├── requirements.txt            # Project-wide dependencies
└── README.md                  # This file
```

## 🔧 Features

### Model Capabilities
- **Characteristic-based Feature Extraction**: Uses Claude 3.5 Sonnet to extract features related to 11 specific characteristics
- **NLP Text Analysis**: Comprehensive text preprocessing including sentiment, cohesiveness, and linguistic features
- **XGBoost Classification**: Advanced gradient boosting with proper regularization
- **Explainable AI**: SHAP-based feature importance and contribution analysis
- **Interactive Predictions**: Real-time text analysis and classification

### Web Application Features
- **Model Performance Dashboard**: View accuracy, ROC AUC, and confusion matrix
- **Feature Importance Analysis**: Interactive visualizations of model features
- **Characteristic Analysis**: Detailed breakdown of characteristic contributions
- **Prediction Interface**: Real-time text classification with confidence scores
- **Batch Prediction**: Upload CSV files for bulk analysis
- **Explainability Insights**: Understand model decisions and feature contributions

## 📊 Model Performance

The model achieves reliable accuracy in TD/ASD classification through:
- **Test Accuracy**: ~89%
- **ROC AUC**: ~96%
- **Cross-validation**: Robust performance across folds
- **Explainability**: Comprehensive feature importance analysis

## 🛠️ Technical Details

### Dependencies
- **Core ML**: XGBoost, scikit-learn, SHAP
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: streamlit
- **NLP**: textblob, nltk, textstat
- **Cloud**: boto3 (for Claude 3.5 Sonnet)

### Path Management
All paths now use the `pathlib.Path` package for cross-platform compatibility and cleaner code.

### Target Leakage Prevention
- Fixed double train-test splits
- Enhanced feature selection
- Prevented feature extraction leakage
- Improved regularization

## 🚀 Usage

### Training the Model
```bash
cd ANN-Modeling/Code/Model_V1
python train.py
```

### Making Predictions
```bash
python predict.py
```

### Running the Web App
```bash
cd ../Demo
streamlit run app.py
```

### Single Text Prediction
```python
from predict import predict_single_sample

result = predict_single_sample("Sample text here", "subject_id")
print(f"Prediction: {'ASD' if result['predicted_td_or_asd'] == 1 else 'TD'}")
print(f"Confidence: {result['prediction_confidence']:.3f}")
```

## 📈 Results

All results are saved to `ANN-Modeling/Results/V2/` including:
- Trained model files
- Feature importance analysis
- Explainability reports
- Comprehensive visualizations
- Prediction results

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