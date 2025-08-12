import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def add_pe_features_to_preprocessed_data():
    """
    Add PE features to existing preprocessed data files that already contain 
    Sonnet-processed characteristic features, avoiding re-running feature extraction.
    """
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Load original data to get PE features
    original_data_path = project_root / "data" / "Data_v1" / "LLM data.csv"
    original_df = pd.read_csv(original_data_path)
    print(f"Loaded original data with shape: {original_df.shape}")
    
    # Instead of using mismatched files, recreate the proper V3 split with PE features
    # Filter the original data to include PE
    selected_columns = ['FSR', 'avg_PE', 'free_response', 'td_or_asd']
    df_filtered = original_df[selected_columns].copy()
    df_filtered = df_filtered.dropna(subset=['FSR', 'avg_PE', 'free_response'])
    
    print(f"Filtered data with PE features shape: {df_filtered.shape}")
    
    # Use train_test_split with the same random_state as the original preprocessing
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df_filtered, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_filtered['td_or_asd']
    )
    
    print(f"New train set shape: {train_df.shape}")
    print(f"New test set shape: {test_df.shape}")
    
    # Now we need to add the characteristic features from existing preprocessed files
    # Load existing preprocessed files to extract characteristic features
    old_train_path = project_root / "data" / "Data_v1" / "LLM_data_train_preprocessed.csv"
    old_test_path = project_root / "data" / "Data_v1" / "LLM_data_test_preprocessed.csv"
    
    if old_train_path.exists() and old_test_path.exists():
        old_train_df = pd.read_csv(old_train_path)
        old_test_df = pd.read_csv(old_test_path)
        
        # Extract characteristic and NLP features (all columns except basic ones)
        basic_cols = ['FSR', 'FSR_scaled', 'free_response', 'td_or_asd']
        feature_cols = [col for col in old_train_df.columns if col not in basic_cols]
        
        print(f"Found {len(feature_cols)} characteristic/NLP features to transfer")
        
        # Create a mapping based on free_response text to match samples
        def create_response_to_features_map(old_df, feature_cols):
            response_map = {}
            for idx, row in old_df.iterrows():
                response_text = str(row['free_response']).strip()
                features = {col: row[col] for col in feature_cols}
                response_map[response_text] = features
            return response_map
        
        # Create mapping from old data
        old_train_map = create_response_to_features_map(old_train_df, feature_cols)
        old_test_map = create_response_to_features_map(old_test_df, feature_cols)
        combined_map = {**old_train_map, **old_test_map}  # Combine both maps
        
        print(f"Created feature mapping for {len(combined_map)} unique responses")
        
        # Function to add features to new data
        def add_features_to_df(new_df, feature_map, feature_cols, dataset_name):
            print(f"\nProcessing {dataset_name} data...")
            result_df = new_df.copy()
            
            # Scale FSR and PE
            scaler_fsr = StandardScaler()
            scaler_pe = StandardScaler()
            result_df['FSR_scaled'] = scaler_fsr.fit_transform(result_df[['FSR']])
            result_df['avg_PE_scaled'] = scaler_pe.fit_transform(result_df[['avg_PE']])
            
            # Initialize feature columns
            for col in feature_cols:
                result_df[col] = 0.0
            
            # Match and add features
            matched_count = 0
            for idx, row in result_df.iterrows():
                response_text = str(row['free_response']).strip()
                if response_text in feature_map:
                    for col, value in feature_map[response_text].items():
                        result_df.loc[idx, col] = value
                    matched_count += 1
            
            print(f"Matched features for {matched_count}/{len(result_df)} samples in {dataset_name}")
            print(f"{dataset_name} data shape after adding features: {result_df.shape}")
            return result_df
        
        # Add features to both datasets
        train_with_pe = add_features_to_df(train_df, combined_map, feature_cols, "Train")
        test_with_pe = add_features_to_df(test_df, combined_map, feature_cols, "Test")
    else:
        print("ERROR: Original preprocessed files not found!")
        return None, None
    
    # Save the updated files with V3 suffix
    train_output_path = project_root / "data" / "Data_v1" / "LLM_data_train_preprocessed_v3.csv"
    test_output_path = project_root / "data" / "Data_v1" / "LLM_data_test_preprocessed_v3.csv"
    
    train_with_pe.to_csv(train_output_path, index=False)
    test_with_pe.to_csv(test_output_path, index=False)
    
    print(f"\nUpdated files saved:")
    print(f"Train: {train_output_path}")
    print(f"Test: {test_output_path}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Original train features: {train_df.shape[1]}")
    print(f"Updated train features: {train_with_pe.shape[1]}")
    print(f"Original test features: {test_df.shape[1]}")
    print(f"Updated test features: {test_with_pe.shape[1]}")
    
    return train_with_pe, test_with_pe

if __name__ == "__main__":
    train_data, test_data = add_pe_features_to_preprocessed_data()