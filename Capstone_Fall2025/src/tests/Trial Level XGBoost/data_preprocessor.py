import pandas as pd
import numpy as np
import os
import json
import sys
import time
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import nltk
from textblob import TextBlob
import textstat
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import boto3
from botocore.exceptions import ClientError
import re
from tqdm import tqdm



def preprocess_training_data():
    current_dir = Path(__file__).parent

    train_base = current_dir / "LLM_data_train_preprocessed_V5.csv"
    test_base = current_dir / "LLM_data_test_preprocessed_V5.csv"
    trial_base = current_dir / "LLM Trial Level Data.csv"

    print(f"Loading data from: {train_base}")
    df_tr = pd.read_csv(train_base)
    df_tr.drop(columns=['mean_slope_overall_cat','mean_slope_overall_subcat','mean_slope_subcat','mean_slope_cat','avg_PE','avg_PE_scaled'], inplace=True)

    print(f"Loading data from: {test_base}")
    df_ts = pd.read_csv(test_base)
    df_ts.drop(columns=['mean_slope_overall_cat', 'mean_slope_overall_subcat', 'mean_slope_subcat', 'mean_slope_cat','avg_PE','avg_PE_scaled'],
               inplace=True)

    print(f"Loading Trial Level Data from : {trial_base}")
    df_trial = pd.read_csv(trial_base)
    selected_cols = ['sub','profile','trial','PE']
    df_trial = df_trial[selected_cols]


    print("Merging Trial Level data with Base Dataset:")
    train_df = pd.merge(df_trial, df_tr, on=["sub", "profile"], how="inner")
    test_df = pd.merge(df_trial, df_ts, on=["sub", "profile"], how="inner")

    print(train_df.columns)
    train_df.to_csv(current_dir / 'train.csv', index=False)
    test_df.to_csv(current_dir / 'test.csv', index=False)

    return train_df, test_df

def preprocess_prediction_data(df,is_test_data=False):
    current_dir = Path(__file__).parent
    df = pd.read_csv(current_dir / 'test.csv')
    return df


if __name__ == "__main__":
    processed_data = preprocess_training_data()