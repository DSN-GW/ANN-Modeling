# Autism Classification using Transformer-based Model

## Project Overview
This project implements a machine learning solution for classifying autism spectrum disorder (ASD) based on a combination of text and numerical features. The model uses a pretrained transformer architecture for processing text data, combined with numerical features, to predict autism diagnosis.

## Dataset Description
The dataset contains information about individuals with and without autism spectrum disorder. It includes:

- Text data: Free-response descriptions of behaviors and preferences
- Numerical features: Various scores and measurements related to autism assessment
- Target variable: Binary classification of autism (td_or_asd)

### Features Used
- **Text Features**: Free-response text descriptions of behaviors and preferences
- **Numerical Features**: 
  - SRS.Raw: Social Responsiveness Scale raw score
  - FSR: Feature score
  - BIS: Behavioral Inhibition Scale score
  - avg_PE: Average Prediction Error
  - LPA_Profile_grand_mean: Latent Profile Analysis grand mean
  - LPA_Profile_ASD_only: Latent Profile Analysis ASD-specific score

## Model Architecture

The model implements a sophisticated multi-modal architecture that combines textual and numerical features for autism spectrum disorder classification. The architecture leverages transfer learning from biomedical domain knowledge while incorporating domain-specific numerical assessments.

### Core Components

#### 1. Text Processing Pipeline
- **Transformer Backbone**: Microsoft BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  - Pre-trained on biomedical literature for domain-specific understanding
  - 768-dimensional hidden representations
  - Partial fine-tuning strategy: First 6 layers and embeddings frozen for stability
- **Text Projection Network**: Multi-layer dimensionality reduction
  - Layer 1: 768 → 256 (BatchNorm + ReLU + 30% Dropout)
  - Layer 2: 256 → 128 (BatchNorm + ReLU + 20% Dropout)  
  - Layer 3: 128 → 64 (BatchNorm + ReLU + 10% Dropout)
  - Output: 64-dimensional text feature representation

#### 2. Numerical Processing Pipeline
- **Input Features**: 6 clinical assessment scores
  - SRS.Raw, FSR, BIS, avg_PE, LPA_Profile_grand_mean, LPA_Profile_ASD_only
- **Numerical Projection Network**: Deep feature extraction
  - Layer 1: 6 → 32 (BatchNorm + ReLU + 20% Dropout)
  - Layer 2: 32 → 64 (BatchNorm + ReLU + 20% Dropout)
  - Layer 3: 64 → 32 (BatchNorm + ReLU + 10% Dropout)
  - Output: 32-dimensional numerical feature representation

#### 3. Feature Fusion and Classification
- **Feature Concatenation**: Combines 64-dim text + 32-dim numerical = 96-dim
- **Multi-head Attention**: 4-head attention mechanism for feature interaction
- **Feature Fusion Network**: 
  - Input: 96 → 128 (BatchNorm + ReLU + 30% Dropout)
  - Residual connection when dimensions match
- **Final Classifier**: Multi-layer classification head
  - Layer 1: 128 → 64 (BatchNorm + ReLU + 20% Dropout)
  - Layer 2: 64 → 32 (BatchNorm + ReLU + 10% Dropout)
  - Output: 32 → 1 (Binary classification logit)

### Detailed Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                TEXT PROCESSING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Text Input (free_response)                                                     │
│           │                                                                     │
│           ▼                                                                     │
│  ┌─────────────────┐    Tokenization & Encoding                                │
│  │  PubMedBERT     │    • Max length: 128 tokens                               │
│  │  Tokenizer      │    • Padding & truncation                                 │
│  └─────────┬───────┘    • Return tensors: PyTorch                              │
│            │                                                                    │
│            ▼                                                                    │
│  ┌─────────────────┐    Transfer Learning Strategy                             │
│  │  PubMedBERT     │    • Embeddings: FROZEN                                   │
│  │  Transformer    │    • Layers 0-5: FROZEN                                   │
│  │  (768-dim)      │    • Layers 6-11: TRAINABLE                               │
│  └─────────┬───────┘    • CLS token extraction                                 │
│            │                                                                    │
│            ▼                                                                    │
│  ┌─────────────────┐                                                           │
│  │ Text Projection │    768 → 256 → 128 → 64                                  │
│  │   Network       │    BatchNorm + ReLU + Dropout                             │
│  │   (64-dim)      │    Dropout rates: 30%, 20%, 10%                          │
│  └─────────┬───────┘                                                           │
└─────────────┼─────────────────────────────────────────────────────────────────┘
              │
              │
┌─────────────┼─────────────────────────────────────────────────────────────────┐
│             │                NUMERICAL PROCESSING PIPELINE                    │
├─────────────┼─────────────────────────────────────────────────────────────────┤
│             │                                                                 │
│             │    ┌─────────────────┐                                          │
│             │    │ Numerical Input │   6 Clinical Features:                   │
│             │    │   (6 features)  │   • SRS.Raw, FSR, BIS                    │
│             │    └─────────┬───────┘   • avg_PE, LPA_Profile_grand_mean       │
│             │              │           • LPA_Profile_ASD_only                 │
│             │              ▼                                                   │
│             │    ┌─────────────────┐   Preprocessing:                         │
│             │    │ StandardScaler  │   • Mean imputation for missing values   │
│             │    │ Normalization   │   • Z-score standardization              │
│             │    └─────────┬───────┘                                          │
│             │              │                                                   │
│             │              ▼                                                   │
│             │    ┌─────────────────┐                                          │
│             │    │   Numerical     │   6 → 32 → 64 → 32                      │
│             │    │   Projection    │   BatchNorm + ReLU + Dropout             │
│             │    │   (32-dim)      │   Dropout rates: 20%, 20%, 10%          │
│             │    └─────────┬───────┘                                          │
└─────────────────────────────┼─────────────────────────────────────────────────┘
                              │
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE FUSION & CLASSIFICATION                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│         ┌─────────────────────────────────────┐                                │
│         │        Feature Concatenation        │   64-dim + 32-dim = 96-dim     │
│         │             (96-dim)                │                                 │
│         └────────────────┬───────────────────┘                                │
│                          │                                                     │
│                          ▼                                                     │
│         ┌─────────────────────────────────────┐                                │
│         │      Multi-Head Attention           │   4 heads, 96-dim embeddings   │
│         │        (Self-Attention)             │   Feature interaction learning │
│         └────────────────┬───────────────────┘                                │
│                          │                                                     │
│                          ▼                                                     │
│         ┌─────────────────────────────────────┐                                │
│         │       Feature Fusion Network        │   96 → 128                     │
│         │         (128-dim)                   │   BatchNorm + ReLU + 30% Drop  │
│         └────────────────┬───────────────────┘                                │
│                          │                                                     │
│                          ▼                                                     │
│         ┌─────────────────────────────────────┐                                │
│         │      Classification Head            │   128 → 64 → 32 → 1           │
│         │                                     │   BatchNorm + ReLU + Dropout   │
│         │                                     │   Dropout: 20%, 10%            │
│         └────────────────┬───────────────────┘                                │
│                          │                                                     │
│                          ▼                                                     │
│         ┌─────────────────────────────────────┐                                │
│         │           Binary Output             │   Sigmoid activation           │
│         │      (Autism vs Non-Autism)         │   BCEWithLogitsLoss            │
│         └─────────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Features
- **Partial Fine-tuning**: Strategic freezing of early transformer layers to prevent overfitting
- **Multi-scale Processing**: Different projection depths for text (3 layers) vs numerical (3 layers) features
- **Regularization Strategy**: Progressive dropout reduction through network depth
- **Feature Interaction**: Multi-head attention mechanism for learning cross-modal relationships
- **Residual Connections**: Skip connections where dimensionally feasible for gradient flow

## Implementation Details

## Data Splitting and Class Imbalance Handling

### Dataset Characteristics
The dataset contains **2,907 samples** after preprocessing, with the following characteristics:

#### Class Distribution
- **Total Samples**: 2,907 individuals
- **Class 0 (Typically Developing - TD)**: ~37% of dataset
- **Class 1 (Autism Spectrum Disorder - ASD)**: ~63% of dataset
- **Imbalance Ratio**: Approximately 1.7:1 (ASD:TD)

#### Sample Composition
The dataset includes diverse behavioral profiles:
- **ASDprof_norm**: Normalized ASD behavioral profiles
- **ASDprof_unif**: Uniform ASD behavioral profiles  
- **TDprof_norm**: Normalized typically developing profiles
- **Text Length**: Variable free-response descriptions (ranging from brief phrases to detailed paragraphs)
- **Missing Data**: Some numerical features contain missing values, handled through mean imputation

### Data Splitting Strategy

#### Split Configuration
```
Training Set: 80% (~2,326 samples)
Validation Set: 20% (~581 samples)
```

#### Stratified Splitting
- **Method**: `train_test_split` with `stratify=y`
- **Purpose**: Maintains proportional class distribution across train/validation splits
- **Random State**: 42 (for reproducibility)
- **Validation Results**: 
  - Class 0 (TD): 196 samples (33.7%)
  - Class 1 (ASD): 334 samples (57.5%)
  - Total validation: 530 samples

### Class Imbalance Mitigation

#### Weighted Random Sampling
The training process addresses class imbalance through **WeightedRandomSampler**:

```python
# Class weight calculation
class_counts = np.bincount(labels.astype(int))
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels.astype(int)]
```

#### Sampling Strategy
- **Inverse Frequency Weighting**: Minority class (TD) samples are upsampled during training
- **Replacement**: `replacement=True` allows repeated sampling of minority class
- **Batch Composition**: Each training batch maintains balanced representation
- **Effect**: Prevents model bias toward majority class (ASD)

#### Alternative Approaches Considered
- **Class Weights in Loss Function**: Could be applied to BCEWithLogitsLoss
- **SMOTE**: Not used due to mixed data types (text + numerical)
- **Undersampling**: Avoided to preserve valuable data

### Data Quality Assurance

#### Missing Value Handling
- **Text Data**: Missing free-response entries filled with empty strings
- **Numerical Features**: Missing values imputed with feature-wise mean
- **Target Variable**: Samples with missing labels removed from dataset

#### Feature Preprocessing
- **Text**: Tokenized with PubMedBERT tokenizer (max_length=128)
- **Numerical**: StandardScaler normalization (zero mean, unit variance)
- **Validation**: No data leakage - scaler fitted only on training data

### Cross-Validation Considerations
- **Current Approach**: Single train/validation split
- **Future Enhancement**: K-fold cross-validation could provide more robust evaluation
- **Stratification**: Maintains class balance across all folds when implemented

## Model Parameters and Configuration

This section provides a comprehensive overview of all hyperparameters, architectural configurations, and training settings used in the autism classification model.

### Training Parameters

| Parameter | Value | Description | Rationale |
|-----------|-------|-------------|-----------|
| **Epochs** | 20 | Number of training iterations through entire dataset | Sufficient for convergence without overfitting |
| **Batch Size** | 16 | Number of samples processed simultaneously | Balance between memory efficiency and gradient stability |
| **Learning Rate** | 2e-5 | Step size for parameter updates | Standard for transformer fine-tuning |
| **Weight Decay** | 0.01 | L2 regularization coefficient | Prevents overfitting in AdamW optimizer |
| **Optimizer** | AdamW | Adaptive learning rate optimization | Superior performance for transformer models |
| **Loss Function** | BCEWithLogitsLoss | Binary cross-entropy with logits | Numerically stable for binary classification |
| **Random State** | 42 | Seed for reproducible data splitting | Ensures consistent train/validation splits |

### Learning Rate Scheduling

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Scheduler** | ReduceLROnPlateau | Adaptive learning rate reduction |
| **Mode** | min | Monitors validation loss decrease |
| **Factor** | 0.5 | Learning rate reduction factor |
| **Patience** | 3 | Epochs to wait before reduction |
| **Min LR** | Not specified | Minimum learning rate threshold |

### Model Architecture Parameters

#### Transformer Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model Name** | microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext | Pre-trained transformer model |
| **Hidden Size** | 768 | Transformer hidden dimension |
| **Max Sequence Length** | 128 | Maximum input token length |
| **Frozen Layers** | Embeddings + Layers 0-5 | Layers with frozen parameters |
| **Trainable Layers** | Layers 6-11 | Fine-tuned transformer layers |

#### Text Processing Network
| Layer | Input Dim | Output Dim | Activation | Dropout | Batch Norm |
|-------|-----------|------------|------------|---------|------------|
| **Text Proj 1** | 768 | 256 | ReLU | 30% | Yes |
| **Text Proj 2** | 256 | 128 | ReLU | 20% | Yes |
| **Text Proj 3** | 128 | 64 | ReLU | 10% | Yes |

#### Numerical Processing Network
| Layer | Input Dim | Output Dim | Activation | Dropout | Batch Norm |
|-------|-----------|------------|------------|---------|------------|
| **Num Proj 1** | 6 | 32 | ReLU | 20% | Yes |
| **Num Proj 2** | 32 | 64 | ReLU | 20% | Yes |
| **Num Proj 3** | 64 | 32 | ReLU | 10% | Yes |

#### Feature Fusion and Classification
| Component | Configuration | Description |
|-----------|---------------|-------------|
| **Multi-Head Attention** | 4 heads, 96-dim embeddings | Cross-modal feature interaction |
| **Feature Fusion** | 96 → 128, ReLU, 30% dropout | Combined feature processing |
| **Classifier Layer 1** | 128 → 64, ReLU, 20% dropout | Classification head |
| **Classifier Layer 2** | 64 → 32, ReLU, 10% dropout | Intermediate classification |
| **Output Layer** | 32 → 1, No activation | Binary logit output |

### Data Preprocessing Parameters

#### Text Processing
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Tokenizer** | PubMedBERT tokenizer | Biomedical domain tokenization |
| **Padding** | True | Pad sequences to max length |
| **Truncation** | True | Truncate sequences exceeding max length |
| **Max Length** | 128 | Maximum token sequence length |
| **Return Tensors** | PyTorch | Output tensor format |
| **Missing Text Handling** | Empty string fill | Default for missing free_response |

#### Numerical Processing
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Scaler** | StandardScaler | Z-score normalization |
| **Missing Value Strategy** | Mean imputation | Feature-wise mean replacement |
| **Feature Count** | 6 | Number of numerical features |
| **Features** | SRS.Raw, FSR, BIS, avg_PE, LPA_Profile_grand_mean, LPA_Profile_ASD_only | Clinical assessment scores |

### Data Splitting Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Train Split** | 80% | Training data proportion |
| **Validation Split** | 20% | Validation data proportion |
| **Stratification** | True | Maintain class distribution |
| **Shuffle** | True | Randomize sample order |
| **Random State** | 42 | Reproducibility seed |

### Class Imbalance Handling

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sampling Strategy** | WeightedRandomSampler | Inverse frequency weighting |
| **Replacement** | True | Allow repeated sampling |
| **Weight Calculation** | 1.0 / class_counts | Inverse class frequency |
| **Class Ratio (ASD:TD)** | ~1.7:1 | Original dataset imbalance |

### Hardware and Environment

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Device Selection** | CUDA if available, else CPU | Automatic GPU utilization |
| **Mixed Precision** | Not specified | Could be enabled for efficiency |
| **Gradient Accumulation** | 1 | No gradient accumulation used |
| **Parallel Processing** | Single GPU/CPU | No multi-device training |

### Model Selection and Evaluation

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Selection Metric** | Training F1 Score | Best model selection criterion |
| **Evaluation Metrics** | Accuracy, F1, AUC | Comprehensive performance assessment |
| **Threshold** | 0.5 | Binary classification threshold |
| **Model Persistence** | PyTorch state_dict | Saved model format |

### Regularization Strategy

| Technique | Configuration | Purpose |
|-----------|---------------|---------|
| **Dropout** | Progressive: 30% → 20% → 10% | Prevent overfitting through network depth |
| **Batch Normalization** | All hidden layers | Stabilize training and improve convergence |
| **Weight Decay** | 0.01 in AdamW | L2 regularization |
| **Layer Freezing** | First 6 transformer layers | Prevent catastrophic forgetting |
| **Early Stopping** | Based on training F1 | Prevent overfitting |

### Model Training
- The model is trained for 20 epochs with a batch size of 16
- AdamW optimizer with a learning rate of 2e-5 and weight decay of 0.01
- ReduceLROnPlateau scheduler with factor 0.5 and patience 3
- Binary Cross-Entropy with Logits Loss function
- Model selection based on best training F1 score

## Evaluation Metrics
The model is evaluated using the following metrics:
- Accuracy: Proportion of correct predictions
- F1 Score: Harmonic mean of precision and recall
- AUC (Area Under the ROC Curve): Measures the model's ability to discriminate between classes

### Visualizations
- ROC Curve: Shows the trade-off between true positive rate and false positive rate
- Precision-Recall Curve: Shows the trade-off between precision and recall
- Confusion Matrix: Shows the counts of true positives, false positives, true negatives, and false negatives

## Usage Instructions

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

### Running the Code
1. Clone the repository
2. Ensure the data is in the `data/Data_v1` directory
3. Run the main script:
   ```
   cd Code
   python main.py
   ```
4. Results will be saved in the `Results/V1` directory

## Results
The model achieves competitive performance on the autism classification task. Detailed metrics and visualizations are saved in the Results/V1 directory after running the code.

