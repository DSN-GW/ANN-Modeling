import os
import pandas as pd
from data_preprocessor import preprocess_training_data
from xgboost_model import train_xgboost_model


def run_complete_training_pipeline(batch_size=10):
    print("=" * 60)
    print("STARTING MODEL V1 TRAINING PIPELINE")
    print("=" * 60)

    try:
        print("\nSTEP 1: Data Preprocessing with Stratification")
        print("-" * 30)
        # Check if preprocessed training and test data exist (using _v1 files for Model V1)
        train_preprocessed_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_train_preprocessed_v1.csv')
        test_preprocessed_path = os.path.join('..', '..', 'data', 'Data_v1', 'LLM_data_test_preprocessed_v1.csv')
        artifacts_dir = os.path.join('..', '..', 'Results', 'V1', 'preprocessing')
        
        if (not os.path.exists(train_preprocessed_path) or 
            not os.path.exists(test_preprocessed_path) or 
            not os.path.exists(artifacts_dir)):
            print("Preprocessed data or artifacts not found. Running stratified preprocessing...")
            print(f"Using batch size: {batch_size} for faster processing")
            train_data, test_data = preprocess_training_data(batch_size=batch_size)
            print(f"Training data preprocessing completed. Shape: {train_data.shape}")
            print(f"Test data preprocessing completed. Shape: {test_data.shape}")
        else:
            print("Preprocessed training and test data already exist. Loading existing data...")
            train_data = pd.read_csv(train_preprocessed_path)
            test_data = pd.read_csv(test_preprocessed_path)
            print(f"Loaded preprocessed training data. Shape: {train_data.shape}")
            print(f"Loaded preprocessed test data. Shape: {test_data.shape}")

        print("\nSTEP 2: Model Training")
        print("-" * 30)
        classifier, training_results = train_xgboost_model()
        print("Model training completed successfully.")

        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"\nModel saved to: ANN-Modeling/Results/V1/")
        print("Run predict.py to perform analysis and visualization on test data.")

        return {
            'classifier': classifier,
            'training_results': training_results
        }

    except Exception as e:
        print(f"\nERROR in training pipeline: {str(e)}")
        print("Training pipeline failed.")
        return None


if __name__ == "__main__":
    # You can adjust batch_size based on your needs
    # Larger batch_size = faster processing but more memory usage
    # Smaller batch_size = slower processing but less memory usage
    batch_size = 10  # Process 10 samples at a time
    results = run_complete_training_pipeline(batch_size=batch_size)