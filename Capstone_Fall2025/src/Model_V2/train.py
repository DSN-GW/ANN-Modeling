import os
import sys
import pandas as pd
from data_preprocessor import preprocess_training_data
from xgboost_model import train_xgboost_model

data_version = "Data_v2"
model_version = "V2"

def run_complete_training_pipeline():
    print("=" * 60)
    print("STARTING MODEL V2 TRAINING PIPELINE")
    print("=" * 60)

    try:
        print("\nSTEP 1: Data Preprocessing")
        print("-" * 30)
        
        train_preprocessed_path = os.path.join('..', '..', 'data', data_version, f'LLM_data_train_preprocessed_{model_version}.csv')
        test_preprocessed_path = os.path.join('..', '..', 'data', data_version, f'LLM_data_test_preprocessed_{model_version}.csv')
        
        if (not os.path.exists(train_preprocessed_path) or 
            not os.path.exists(test_preprocessed_path)):
            print("Preprocessed data not found. Running preprocessing...")
            train_data, test_data = preprocess_training_data()
            print(f"Training data preprocessing completed. Shape: {train_data.shape}")
            print(f"Test data preprocessing completed. Shape: {test_data.shape}")
        else:
            print("Preprocessed data already exists. Loading existing data...")
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

        print(f"\nModel saved to: ANN-Modeling/Results/{model_version}/")
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
    results = run_complete_training_pipeline()