# Model V2 - Advanced TD/ASD Classification System

## Overview
Advanced machine learning system for classifying Typically Developing (TD) vs Autism Spectrum Disorder (ASD) individuals based on free response text analysis using XGBoost and comprehensive NLP features.

## Features
- Characteristic-based feature extraction using Claude 3.5 Sonnet
- Comprehensive NLP text preprocessing and feature engineering
- XGBoost classification with explainability analysis
- Interactive Streamlit demo application
- Comprehensive visualization and analysis tools

## Project Structure
```
Model_V2/
├── feature_extractor.py      # Characteristic-based feature extraction
├── nlp_features.py          # NLP text preprocessing features
├── data_preprocessor.py     # Data preprocessing pipeline
├── xgboost_model.py         # XGBoost model training and evaluation
├── explainability_analysis.py # SHAP-based explainability analysis
├── visualization.py         # Comprehensive visualization generation
├── train.py                 # Complete training pipeline
├── predict.py               # Prediction pipeline with visualization
└── README.md               # This file
```

## Usage

### Training the Model
```bash
cd ANN-Modeling/Code/Model_V2
python train.py
```

### Making Predictions
```bash
cd ANN-Modeling/Code/Model_V2
python predict.py
```

### Running the Demo
```bash
cd ANN-Modeling/Code/Demo
streamlit run app.py
```

## Model Pipeline
1. **Data Preprocessing**: Extract characteristic and NLP features from free response text
2. **Model Training**: Train XGBoost classifier with comprehensive feature set
3. **Explainability Analysis**: Generate SHAP-based feature importance and contribution analysis
4. **Visualization**: Create comprehensive analysis charts and plots
5. **Prediction**: Make predictions on new data with confidence scores

## Results
All results are saved to `ANN-Modeling/Results/V2/` including:
- Trained model files
- Feature importance analysis
- Explainability reports
- Comprehensive visualizations
- Prediction results

## Requirements
- Python 3.8+
- XGBoost
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- plotly
- SHAP
- textblob
- nltk
- textstat
- boto3 (for Claude 3.5 Sonnet integration)

## Model Performance
The model achieves reliable accuracy in TD/ASD classification through:
- Multi-dimensional feature extraction from text
- Advanced gradient boosting classification with proper regularization
- Comprehensive explainability analysis
- Interactive prediction interface
- Careful prevention of target leakage and overfitting

## Target Leakage Analysis
A comprehensive analysis was performed to identify and fix potential target leakage and overfitting issues in the model. See [target_leakage_analysis.md](target_leakage_analysis.md) for detailed findings and fixes.

### Key Improvements
1. **Fixed Double Train-Test Split**: Eliminated redundant train-test splits to ensure evaluation on truly unseen data
2. **Enhanced Feature Selection**: Removed features with potential target leakage and implemented correlation-based filtering
3. **Prevented Feature Extraction Leakage**: Ensured separate feature extraction for train and test data
4. **Improved Regularization**: Added proper regularization parameters to prevent overfitting

These improvements result in a more robust and reliable model that generalizes better to unseen data.