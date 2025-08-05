# Model V2 - Advanced TD/ASD Classification System

## Overview
Advanced machine learning system for classifying Typically Developing (TD) vs Autism Spectrum Disorder (ASD) individuals based on free response text analysis. The system combines characteristic-based feature extraction using Claude 3.5 Sonnet with comprehensive NLP features and XGBoost classification to achieve high accuracy and interpretability.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Feature Processing Pipeline](#feature-processing-pipeline)
- [Results and Performance](#results-and-performance)
- [Visualizations](#visualizations)
- [Usage](#usage)
- [Requirements](#requirements)

## Model Architecture

### System Overview
The Model V2 system implements a multi-stage machine learning pipeline designed for robust TD/ASD classification:

```
Raw Text Input
      ↓
┌─────────────────────────────────────┐
│     Data Preprocessing              │
│  • Stratified train-test split     │
│  • Data leakage prevention         │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│    Feature Extraction Pipeline     │
│  ┌─────────────────────────────────┐│
│  │  Characteristic-Based Features │││
│  │  • Claude 3.5 Sonnet Analysis  │││
│  │  • 11 behavioral characteristics│││
│  │  • Sentiment & mention detection│││
│  └─────────────────────────────────┘│
│  ┌─────────────────────────────────┐│
│  │      NLP Features              │││
│  │  • Basic text statistics       │││
│  │  • Sentiment analysis          │││
│  │  • Cohesiveness metrics        │││
│  │  • Linguistic features         │││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│      XGBoost Classifier            │
│  • Gradient boosting algorithm     │
│  • Regularization parameters       │
│  • Cross-validation optimization   │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│    Explainability Analysis         │
│  • SHAP feature importance         │
│  • Characteristic ranking          │
│  • Prediction confidence scores    │
└─────────────────────────────────────┘
```

### Core Components

#### 1. CharacteristicFeatureExtractor
- **Purpose**: Extracts behavioral and preference features using AI analysis
- **Technology**: Claude 3.5 Sonnet (Anthropic's LLM)
- **Features**: 22 features (11 characteristics × 2 aspects each)
- **Processing**: Batch processing with error handling and logging

#### 2. NLPFeatureExtractor  
- **Purpose**: Comprehensive linguistic and statistical text analysis
- **Features**: 29 NLP features across 4 categories
- **Libraries**: NLTK, TextBlob, TextStat

#### 3. XGBoostClassifier
- **Algorithm**: Extreme Gradient Boosting
- **Optimization**: 5-fold stratified cross-validation
- **Regularization**: L1 (0.1) and L2 (1.0) regularization
- **Parameters**: Optimized for generalization and interpretability

### Model Parameters
```python
XGBoost Configuration:
├── n_estimators: 100
├── max_depth: 4
├── learning_rate: 0.05
├── subsample: 0.7
├── colsample_bytree: 0.7
├── gamma: 0.1
├── min_child_weight: 3
├── reg_alpha: 0.1 (L1 regularization)
└── reg_lambda: 1.0 (L2 regularization)
```

## Feature Processing Pipeline

### 1. Characteristic-Based Features (22 features)

The system analyzes 11 behavioral characteristics, extracting 2 features per characteristic:

| Characteristic | Description | Features Extracted |
|---|---|---|
| **Personality Inference** | Social and personality-related content | mentioned, sentiment |
| **Sweets** | Sweet food preferences and mentions | mentioned, sentiment |
| **Fruits and Vegetables** | Healthy food preferences | mentioned, sentiment |
| **Healthy Savory Food** | Nutritious meal preferences | mentioned, sentiment |
| **Food** | General food-related content | mentioned, sentiment |
| **Cosmetics** | Beauty and cosmetic interests | mentioned, sentiment |
| **Fashion** | Clothing and style preferences | mentioned, sentiment |
| **Toys, Gadgets and Games** | Technology and entertainment | mentioned, sentiment |
| **Sports** | Physical activities and sports | mentioned, sentiment |
| **Music** | Musical interests and preferences | mentioned, sentiment |
| **Arts and Crafts** | Creative activities and hobbies | mentioned, sentiment |

**Feature Extraction Process:**
1. **Text Analysis**: Claude 3.5 Sonnet analyzes free response text
2. **Mention Detection**: Binary classification (mentioned/not mentioned)
3. **Sentiment Analysis**: Categorical classification (positive/negative/neutral)
4. **Batch Processing**: Efficient processing with error handling
5. **Quality Control**: Logging and validation of extracted features

### 2. NLP Features (29 features)

#### Basic Text Statistics (8 features)
- `word_count`: Total number of words
- `sentence_count`: Total number of sentences  
- `char_count`: Total character count
- `avg_word_length`: Average word length
- `avg_sentence_length`: Average sentence length
- `shortness_score`: Text brevity metric
- `lexical_diversity`: Vocabulary richness (unique words/total words)

#### Sentiment Analysis (8 features)
- `sentiment_polarity`: Overall emotional tone (-1 to 1)
- `sentiment_subjectivity`: Subjectivity score (0 to 1)
- `positive_word_count`: Count of positive words
- `negative_word_count`: Count of negative words
- `positive_word_ratio`: Ratio of positive words
- `negative_word_ratio`: Ratio of negative words
- `positive_attributes`: Positive characteristic mentions

#### Cohesiveness Features (4 features)
- `flesch_reading_ease`: Text readability score
- `flesch_kincaid_grade`: Grade level readability
- `connector_ratio`: Ratio of connecting words
- `cohesiveness_score`: Overall text coherence

#### Linguistic Features (9 features)
- `noun_ratio`: Proportion of nouns
- `verb_ratio`: Proportion of verbs
- `adj_ratio`: Proportion of adjectives
- `adv_ratio`: Proportion of adverbs
- `punctuation_ratio`: Punctuation density
- `exclamation_count`: Number of exclamation marks
- `question_count`: Number of question marks

### 3. Data Preprocessing Pipeline

#### Train-Test Split Strategy
```python
# Stratified split to maintain class balance
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['td_or_asd']
)
```

#### Data Leakage Prevention
- **Separate Feature Extractors**: Independent extractors for train/test data
- **No Information Leakage**: Feature extraction performed separately on each split
- **Validation Strategy**: Proper cross-validation within training set only

## Results and Performance

### Model Performance Metrics

#### Cross-Validation Results (Training)
| Metric | Value |
|---|---|
| **Mean CV Accuracy** | **92.54%** |
| **Standard Deviation** | **0.55%** |
| **CV Scores** | [91.75%, 92.69%, 92.67%, 92.20%, 93.38%] |

#### Test Set Performance
| Metric | Value |
|---|---|
| **Test Accuracy** | **89.06%** |
| **ROC-AUC Score** | **96.16%** |
| **Precision (TD)** | **85.20%** |
| **Recall (TD)** | **85.20%** |
| **F1-Score (TD)** | **85.20%** |
| **Precision (ASD)** | **91.32%** |
| **Recall (ASD)** | **91.32%** |
| **F1-Score (ASD)** | **91.32%** |

#### Confusion Matrix
```
                Predicted
                TD    ASD
Actual    TD   167    29
         ASD    29   305
```

#### Prediction Confidence Analysis
| Metric | Value |
|---|---|
| **Total Test Samples** | 530 |
| **Average Confidence** | **89.97%** |
| **High Confidence Predictions** | 423 (79.8%) |
| **Low Confidence Predictions** | 34 (6.4%) |
| **TD Predictions** | 196 (37.0%) |
| **ASD Predictions** | 334 (63.0%) |

### Feature Importance Analysis

#### Top 15 Most Important Features
| Rank | Feature | Importance Score | Category |
|---|---|---|---|
| 1 | **FSR** | 21.73% | Clinical |
| 2 | **BIS** | 16.32% | Clinical |
| 3 | **sweets_mentioned** | 3.20% | Characteristic |
| 4 | **positive_attributes** | 3.10% | NLP |
| 5 | **shortness_score** | 2.58% | NLP |
| 6 | **char_count** | 2.43% | NLP |
| 7 | **LPA_Profile_grand_mean** | 2.34% | Clinical |
| 8 | **word_count** | 2.21% | NLP |
| 9 | **toys_gadgets_games_sentiment** | 2.07% | Characteristic |
| 10 | **connector_ratio** | 1.72% | NLP |
| 11 | **verb_ratio** | 1.69% | NLP |
| 12 | **flesch_reading_ease** | 1.69% | NLP |
| 13 | **sports_sentiment** | 1.65% | Characteristic |
| 14 | **positive_word_count** | 1.65% | NLP |
| 15 | **fruits_vegetables_sentiment** | 1.65% | Characteristic |

#### Characteristic Importance Ranking
| Rank | Characteristic | Total Importance | Description |
|---|---|---|---|
| 1 | **Sweets** | 4.20% | Sweet food preferences |
| 2 | **Fruits and Vegetables** | 3.11% | Healthy food choices |
| 3 | **Toys, Gadgets and Games** | 3.06% | Technology/entertainment |
| 4 | **Food** | 2.43% | General food mentions |
| 5 | **Sports** | 2.43% | Physical activities |
| 6 | **Arts and Crafts** | 2.16% | Creative activities |
| 7 | **Personality Inference** | 2.16% | Social/personality content |
| 8 | **Healthy Savory Food** | 1.65% | Nutritious meals |
| 9 | **Music** | 0.68% | Musical interests |
| 10 | **Cosmetics** | 0.00% | Beauty products |
| 11 | **Fashion** | 0.00% | Clothing/style |

### Model Interpretability

#### Key Findings
1. **Clinical Features Dominate**: FSR and BIS account for 38% of total importance
2. **Food Preferences Matter**: Food-related characteristics show high predictive power
3. **Text Style Indicators**: Writing style features (shortness, word count) are significant
4. **Balanced Feature Usage**: Model uses diverse feature types for robust predictions

#### Explainability Features
- **SHAP Analysis**: Individual prediction explanations
- **Feature Contribution**: Per-sample feature impact analysis
- **Characteristic Grouping**: Aggregated importance by behavioral domains
- **Confidence Scoring**: Prediction reliability assessment

## Visualizations

The system generates comprehensive visualizations for model analysis and interpretation:

### Available Charts
1. **`confusion_matrix_v2.png`** - Test set confusion matrix
2. **`feature_importance_v2.png`** - Top feature importance scores
3. **`characteristic_importance_v2.png`** - Characteristic-level importance analysis
4. **`characteristic_ranking_v2.png`** - Ranked characteristic comparison
5. **`model_performance_v2.png`** - Overall model performance metrics
6. **`td_vs_asd_comparison_v2.png`** - Class-specific feature analysis
7. **`test_prediction_analysis_v2.png`** - Prediction confidence distribution
8. **`sample_feature_contributions_v2.png`** - Individual prediction explanations

### Visualization Features
- **Interactive Elements**: Hover information and detailed tooltips
- **Professional Styling**: Publication-ready charts with clear labeling
- **Comprehensive Coverage**: All aspects of model performance and interpretation
- **Export Ready**: High-resolution PNG format for presentations and papers

## Usage

### Training the Model
```bash
cd ANN-Modeling/Code/Model_V2
python train.py
```

**Training Process:**
1. Data preprocessing with stratified split
2. Characteristic feature extraction using Claude 3.5 Sonnet
3. NLP feature extraction and engineering
4. XGBoost model training with cross-validation
5. Model evaluation and explainability analysis
6. Comprehensive visualization generation

### Making Predictions
```bash
cd ANN-Modeling/Code/Model_V2
python predict.py
```

**Prediction Process:**
1. Load trained model and preprocessing artifacts
2. Extract features from new text data
3. Generate predictions with confidence scores
4. Create explainability analysis for predictions
5. Generate prediction visualizations

### Running the Demo
```bash
cd ANN-Modeling/Code/Demo
streamlit run app.py
```

**Demo Features:**
- Interactive text input interface
- Real-time prediction with confidence scores
- Feature contribution visualization
- Model explanation and interpretation

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
├── requirements.txt         # Python dependencies
└── README.md               # This comprehensive documentation
```

## Requirements

### Core Dependencies
```
Python 3.8+
xgboost>=1.6.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
```

### NLP and Text Processing
```
nltk>=3.7
textblob>=0.17.0
textstat>=0.7.0
```

### Visualization and Analysis
```
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
shap>=0.41.0
```

### AI Integration
```
boto3>=1.20.0  # For Claude 3.5 Sonnet integration
anthropic>=0.3.0
```

### Web Interface
```
streamlit>=1.10.0
```

## Model Robustness and Validation

### Target Leakage Prevention
A comprehensive analysis was performed to identify and eliminate potential target leakage:

#### Key Improvements Implemented
1. **Eliminated Double Train-Test Split**: Fixed redundant splitting to ensure evaluation on truly unseen data
2. **Enhanced Feature Selection**: Removed features with potential target leakage using correlation analysis
3. **Prevented Feature Extraction Leakage**: Separate feature extractors for train and test data
4. **Improved Regularization**: Added proper L1/L2 regularization to prevent overfitting

#### Validation Strategy
- **Stratified Cross-Validation**: 5-fold CV maintaining class balance
- **Independent Test Set**: 20% holdout set never seen during training
- **Feature Importance Validation**: Consistent importance across CV folds
- **Generalization Testing**: Performance monitoring on unseen data

### Model Reliability
- **Consistent Performance**: Low standard deviation (0.55%) across CV folds
- **High Confidence Predictions**: 79.8% of predictions with high confidence
- **Balanced Classification**: Good performance on both TD and ASD classes
- **Interpretable Results**: Clear feature importance and explainability

## Technical Specifications

### Computational Requirements
- **Memory**: Minimum 8GB RAM recommended
- **Processing**: Multi-core CPU for efficient training
- **Storage**: ~500MB for model artifacts and results
- **Network**: Internet connection required for Claude 3.5 Sonnet API

### Performance Characteristics
- **Training Time**: ~10-15 minutes on standard hardware
- **Prediction Time**: <1 second per sample
- **Batch Processing**: Efficient handling of large datasets
- **Scalability**: Linear scaling with dataset size

### Quality Assurance
- **Comprehensive Logging**: Detailed error tracking and debugging
- **Input Validation**: Robust handling of edge cases and missing data
- **Error Recovery**: Graceful handling of API failures and processing errors
- **Reproducibility**: Fixed random seeds for consistent results

## Future Enhancements

### Planned Improvements
1. **Advanced Feature Engineering**: Additional linguistic and psychological features
2. **Ensemble Methods**: Combination with other ML algorithms
3. **Real-time Processing**: Streaming prediction capabilities
4. **Multi-language Support**: Extension to non-English text analysis
5. **Enhanced Explainability**: More detailed SHAP analysis and visualization

### Research Directions
- **Deep Learning Integration**: Transformer-based feature extraction
- **Multimodal Analysis**: Integration of text, audio, and behavioral data
- **Longitudinal Modeling**: Time-series analysis of developmental patterns
- **Personalized Predictions**: Individual-specific model adaptation

---

## Citation
If you use this model in your research, please cite:
```
Model V2: Advanced TD/ASD Classification System
Deep Learning Research Group
2024
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions, issues, or collaboration opportunities, please contact the development team.

---
*Last Updated: August 2024*
*Model Version: 2.0*
*Documentation Version: 1.0*