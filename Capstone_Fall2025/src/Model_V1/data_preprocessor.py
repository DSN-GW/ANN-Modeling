import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from feature_extractor import CharacteristicFeatureExtractor
from nlp_features import NLPFeatureExtractor

def preprocess_training_data(batch_size=10):
    """
    Preprocess training data with stratified train-test split.
    Only run feature extraction if preprocessed files don't exist.
    """
    # Get the project root directory (two levels up from current file)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Check if preprocessed files already exist
    train_output_path = project_root / "data" / "Data_v1" / "LLM_data_train_preprocessed_v1.csv"
    test_output_path = project_root / "data" / "Data_v1" / "LLM_data_test_preprocessed_v1.csv"
    
    if train_output_path.exists() and test_output_path.exists():
        print("Preprocessed training and test data already exist. Loading existing files...")
        print("Skipping feature extraction to avoid unnecessary agent sonnet calls.")
        train_processed = pd.read_csv(train_output_path)
        test_processed = pd.read_csv(test_output_path)
        print(f"Loaded preprocessed training data. Shape: {train_processed.shape}")
        print(f"Loaded preprocessed test data. Shape: {test_processed.shape}")
        return train_processed, test_processed
    
    print("Preprocessed data not found. Running feature extraction...")
    
    data_path = project_root / "data" / "Data_v1" / "LLM data.csv"
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    # Create artifacts directory
    artifacts_dir = project_root / "Results" / "V1" / "preprocessing"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessing info
    preprocessing_info = {
        'original_shape': df.shape,
        'columns': list(df.columns),
        'target_distribution': df['td_or_asd'].value_counts().to_dict()
    }
    
    with open(artifacts_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    # Load preprocessing info for feature extraction
    info_path = artifacts_dir / 'preprocessing_info.json'
    with open(info_path, 'r') as f:
        preprocessing_info = json.load(f)
    
    print("Preprocessing info loaded successfully.")
    
    # Perform stratified train-test split
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['td_or_asd']
    )
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Save train and test splits
    train_path = project_root / "data" / "Data_v1" / "LLM_data_train.csv"
    test_path = project_root / "data" / "Data_v1" / "LLM_data_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    
    # Process training data
    print("\nProcessing training data...")
    train_extractor = CharacteristicFeatureExtractor()
    train_extractor.batch_size = batch_size
    
    train_processed = train_extractor.process_dataset(train_df)
    
    # Add NLP features to training data
    nlp_extractor = NLPFeatureExtractor()
    train_processed = nlp_extractor.add_nlp_features(train_processed)
    
    # Save preprocessed training data
    train_processed.to_csv(train_output_path, index=False)
    print(f"Preprocessed training data saved to: {train_output_path}")
    
    # Process test data with separate extractors (to prevent data leakage)
    print("\nProcessing test data with separate extractors...")
    test_extractor = CharacteristicFeatureExtractor()
    test_extractor.batch_size = batch_size
    
    test_processed = test_extractor.process_dataset(test_df)
    
    # Add NLP features to test data
    test_nlp_extractor = NLPFeatureExtractor()
    test_processed = test_nlp_extractor.add_nlp_features(test_processed)
    
    # Save preprocessed test data
    test_processed.to_csv(test_output_path, index=False)
    print(f"Preprocessed test data saved to: {test_output_path}")
    
    return train_processed, test_processed

def preprocess_prediction_data(df, is_test_data=False):
    """
    Preprocess new data for prediction using saved preprocessing artifacts.
    """
    # Get the project root directory (two levels up from current file)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    if is_test_data:
        # Try to load preprocessed test data first
        test_preprocessed_path = project_root / "data" / "Data_v1" / "LLM_data_test_preprocessed_v1.csv"
        
        if test_preprocessed_path.exists():
            print(f"Loading preprocessed test data from: {test_preprocessed_path}")
            return pd.read_csv(test_preprocessed_path)
        else:
            print("Preprocessed test data not found. Processing test data...")
    
    # Load preprocessing info
    artifacts_dir = project_root / "Results" / "V1" / "preprocessing"
    info_path = artifacts_dir / 'preprocessing_info.json'
    
    if not info_path.exists():
        raise FileNotFoundError(f"Preprocessing info not found at {info_path}. Please run training first.")
    
    with open(info_path, 'r') as f:
        preprocessing_info = json.load(f)
    
    print("Preprocessing info loaded successfully.")
    
    # Extract features using fresh extractors
    print("Creating fresh extractors for prediction data...")
    print("This ensures no information leaks from training to prediction data")
    
    extractor = CharacteristicFeatureExtractor()
    processed_df = extractor.process_dataset(df)
    
    # Add NLP features
    nlp_extractor = NLPFeatureExtractor()
    processed_df = nlp_extractor.add_nlp_features(processed_df)
    
    print(f"Preprocessing completed")
    print(f"Original columns: {len(df.columns)}")
    print(f"Final columns: {len(processed_df.columns)}")
    print(f"Total added features: {len(processed_df.columns) - len(df.columns)}")
    
    return processed_df

if __name__ == "__main__":
    processed_data = preprocess_training_data()