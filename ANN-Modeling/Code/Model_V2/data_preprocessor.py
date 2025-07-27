import pandas as pd
import os
from sklearn.model_selection import train_test_split
from feature_extractor import FeatureExtractor
from nlp_features import NLPFeatureExtractor

def load_data():
    data_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM data.csv')
    df = pd.read_csv(data_path)
    return df

def split_data_stratified(df, test_size=0.2, random_state=42):
    print(f"Splitting data with stratification (test_size={test_size})...")
    if 'td_or_asd' not in df.columns:
        raise ValueError("Target column 'td_or_asd' not found in data")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['td_or_asd'])
    print(f"Train set size: {len(train_df)} samples")
    print(f"Test set size: {len(test_df)} samples")
    print(f"Train target distribution: {train_df['td_or_asd'].value_counts().to_dict()}")
    print(f"Test target distribution: {test_df['td_or_asd'].value_counts().to_dict()}")
    return train_df, test_df

def save_preprocessing_artifacts(artifacts_dir='../../Results/V2/preprocessing'):
    os.makedirs(artifacts_dir, exist_ok=True)
    preprocessing_info = {'preprocessing_completed': True}
    import json
    with open(os.path.join(artifacts_dir, 'preprocessing_info.json'), 'w') as f:
        json.dump(preprocessing_info, f)
    print(f"Preprocessing info saved to: {artifacts_dir}")

def load_preprocessing_artifacts(artifacts_dir='../../Results/V2/preprocessing'):
    info_path = os.path.join(artifacts_dir, 'preprocessing_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Preprocessing info not found in {artifacts_dir}")
    print(f"Preprocessing info loaded from: {artifacts_dir}")
    return True

def preprocess_training_data(test_size=0.2, random_state=42, batch_size=10):
    print("=" * 60)
    print("PREPROCESSING TRAINING DATA WITH STRATIFICATION")
    print("=" * 60)
    print("Loading data...")
    df = load_data()
    print("Splitting data with stratification...")
    train_df, test_df = split_data_stratified(df, test_size=test_size, random_state=random_state)
    train_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_train.csv')
    test_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    
    # IMPORTANT: Process train and test data separately to prevent leakage
    print("\nProcessing training data...")
    print("Initializing characteristic feature extractor for training data...")
    train_char_extractor = FeatureExtractor(batch_size=batch_size)  # Use batched processing
    print("Extracting characteristic features from training data...")
    train_processed = train_char_extractor.process_dataset(train_df)
    print("Initializing NLP feature extractor for training data...")
    train_nlp_extractor = NLPFeatureExtractor()
    print("Extracting NLP features from training data...")
    train_final = train_nlp_extractor.process_dataset(train_processed)
    train_output_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_train_preprocessed.csv')
    train_final.to_csv(train_output_path, index=False)
    
    # Process test data with fresh extractors to prevent any leakage
    print("\nProcessing test data...")
    print("Initializing characteristic feature extractor for test data...")
    test_char_extractor = FeatureExtractor(batch_size=batch_size)  # Use batched processing
    print("Extracting characteristic features from test data...")
    test_processed = test_char_extractor.process_dataset(test_df)
    print("Initializing NLP feature extractor for test data...")
    test_nlp_extractor = NLPFeatureExtractor()  # Create a new instance for test data
    print("Extracting NLP features from test data...")
    test_final = test_nlp_extractor.process_dataset(test_processed)
    test_output_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_test_preprocessed.csv')
    test_final.to_csv(test_output_path, index=False)
    print(f"Preprocessed test data saved to: {test_output_path}")
    
    print("\nNote: Train and test data were processed with separate feature extractors to prevent leakage")
    
    save_preprocessing_artifacts()
    print(f"Preprocessed training data saved to: {train_output_path}")
    print(f"Original columns: {len(train_df.columns)}")
    print(f"Final columns: {len(train_final.columns)}")
    print(f"Total added features: {len(train_final.columns) - len(train_df.columns)}")
    return train_final, test_final

def load_preprocessed_test_data():
    """
    Load preprocessed test data if it exists
    """
    test_preprocessed_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_test_preprocessed.csv')
    if os.path.exists(test_preprocessed_path):
        print(f"Loading preprocessed test data from: {test_preprocessed_path}")
        return pd.read_csv(test_preprocessed_path)
    return None

def preprocess_prediction_data(df, is_test_data=False, batch_size=10):

    print("=" * 60)
    print("PREPROCESSING PREDICTION DATA")
    print("=" * 60)
    
    if is_test_data:
        preprocessed_test = load_preprocessed_test_data()
        if preprocessed_test is not None:
            print("Using preprocessed test data (no feature extraction needed)")
            print("Note: This test data was processed with separate extractors from training data")
            return preprocessed_test
        else:
            print("Preprocessed test data not found, performing live preprocessing...")
    

    print("Loading preprocessing info...")
    load_preprocessing_artifacts()

    # IMPORTANT: Always use fresh extractors for prediction data to prevent leakage
    print("Creating fresh extractors for prediction data...")
    print("This ensures no information leaks from training to prediction data")

    pred_char_extractor = FeatureExtractor(batch_size=batch_size)  # Use batched processing
    pred_nlp_extractor = NLPFeatureExtractor()

    print("Extracting characteristic features...")
    processed_df = pred_char_extractor.process_dataset(df)

    print("Extracting NLP features...")
    final_df = pred_nlp_extractor.process_dataset(processed_df)

    print(f"Preprocessing completed")
    print(f"Original columns: {len(df.columns)}")
    print(f"Final columns: {len(final_df.columns)}")
    print(f"Total added features: {len(final_df.columns) - len(df.columns)}")

    return final_df
def preprocess_data():
    print("Warning: preprocess_data() is deprecated. Use preprocess_training_data() instead.")
    train_data, test_data = preprocess_training_data()
    return train_data

if __name__ == "__main__":
    processed_data = preprocess_data()