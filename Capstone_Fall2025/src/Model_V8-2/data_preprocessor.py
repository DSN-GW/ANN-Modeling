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

# Add the Agent directory to the path
current_dir = Path(__file__).parent
agent_dir = current_dir.parent / "Agent"
sys.path.append(str(agent_dir))

data_version = "Data_v8-2"
model_version = "V8-2"

# from sonnet_agent import SonnetAgent
# from llama_agent import MetaLlamaAgent as SonnetAgent
from qwen_agent import QwenAgent as SonnetAgent
# from mistral_agent import MistralAgent as SonnetAgent
# from trial_data_preprocess import preprocess_trial_data

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def preprocess_training_data():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / data_version / "LLM data_aggregate.csv"

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    # df = df.sample(n=200, random_state=42)
    print(f"Original data shape: {df.shape}")

    # print(f"Loading Trial Level Data :")
    # trial_df = preprocess_trial_data()
    #
    # print("Merging Trial Level data with Base Dataset:")
    # merged_df = pd.merge(df, trial_df, on=["sub", "profile"], how="inner")

    artifacts_dir = project_root / "Results" / model_version / "preprocessing"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    selected_columns = ['sub', 'FSR', 'TDNorm_avg_PE','overall_avg_PE','TDnorm_concept_learning','overall_concept_learning', 'free_response_TDprof_norm', 'td_or_asd', ]
    df_filtered = df[selected_columns].copy()

    df_filtered = df_filtered.dropna()

    preprocessing_info = {
        'original_shape': df.shape,
        'filtered_shape': df_filtered.shape,
        'selected_columns': selected_columns,
        'target_distribution': df_filtered['td_or_asd'].value_counts().to_dict()
    }

    df = df_filtered

    with open(artifacts_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)

    # train_df, test_df = train_test_split(
    #     df_filtered,
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=df_filtered['td_or_asd']
    # )

    class0 = df[df['td_or_asd'] == 0]['FSR']
    class1 = df[df['td_or_asd'] == 1]['FSR']

    min0, max0 = class0.min(), class0.max()
    min1, max1 = class1.min(), class1.max()
    overlap_min = max(min0, min1)
    overlap_max = min(max0, max1)

    if overlap_min <= overlap_max:
        df_test = df[(df['FSR'] >= overlap_min) & (df['FSR'] <= overlap_max)]
        df_train = df[~((df['FSR'] >= overlap_min) & (df['FSR'] <= overlap_max))]
        print(f"Overlap range: [{overlap_min:.4f}, {overlap_max:.4f}]")
        print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")
    else:
        print("No overlap between FSR values of class 0 and class 1.")
        df_train = df.copy()
        df_test = pd.DataFrame(columns=df.columns)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_df = df_train
    test_df = df_test

    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    train_path = project_root / "data" / data_version / f"LLM_data_train_{model_version}.csv"
    test_path = project_root / "data" / data_version / f"LLM_data_test_{model_version}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")

    train_processed = process_features(train_df)
    test_processed = process_features(test_df)

    train_output_path = project_root / "data" / data_version / f"LLM_data_train_preprocessed_{model_version}.csv"
    test_output_path = project_root / "data" / data_version / f"LLM_data_test_preprocessed_{model_version}.csv"

    train_processed.to_csv(train_output_path, index=False)
    test_processed.to_csv(test_output_path, index=False)

    print(f"Preprocessed training data saved to: {train_output_path}")
    print(f"Preprocessed test data saved to: {test_output_path}")

    return train_processed, test_processed


class CharacteristicFeatureExtractor:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.agent = SonnetAgent()

        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent

        log_dir = project_root / "Results" / model_version / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'feature_extraction_errors_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        char_path = project_root / "data" / data_version / "charactristic.txt"
        self.characteristics = self.load_characteristics(char_path)

        self.feature_data = {}
        self.failed_samples = []

        # Initialize and setup the agent
        self.setup_agent()

    def load_characteristics(self, char_path):
        try:
            with open(char_path, 'r', encoding='utf-8') as f:
                characteristics = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(characteristics)} characteristics")
            return characteristics
        except FileNotFoundError:
            print(f"Warning: Characteristics file not found at {char_path}")
            return []

    def setup_agent(self):
        system_prompt = f"""You are an expert text analyzer. Your task is to analyze free response text and extract features related to specific characteristics.

The characteristics to analyze are:
{', '.join(self.characteristics)}

For each text, you need to determine:
1. Which characteristics are mentioned or implied
2. The sentiment/preference for each characteristic (positive, negative, neutral)

Return your analysis as a JSON object with the following structure:
{{
    "characteristic_name": {{
        "mentioned": true/false,
        "sentiment": "positive/negative/neutral"
    }}
}}

Be precise and only mark characteristics as mentioned if there is clear evidence in the text.
IMPORTANT: Always return valid JSON format."""

        self.agent.set_system_prompt(system_prompt)
        self.agent.set_parameters(max_tokens=4000, temperature=0.1)

    def extract_characteristic_features(self, text, sample_id):
        if pd.isna(text) or text == "":
            return self.get_empty_characteristic_features()

        try:
            # Use Sonnet agent for analysis
            response = self.agent.ask(f"Analyze this text: '{text}'")

            # Parse JSON response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in other text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")

            # Convert analysis to features
            features = {}
            for char in self.characteristics:
                char_clean = char.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')

                # Initialize default values
                mentioned = 0
                positive = 0
                negative = 0
                neutral = 1

                # Check if characteristic is in analysis
                char_data = analysis.get(char, analysis.get(char_clean, {}))

                if char_data and isinstance(char_data, dict):
                    if char_data.get('mentioned', False):
                        mentioned = 1
                        sentiment = char_data.get('sentiment', 'neutral').lower()

                        if sentiment == 'positive':
                            positive = 1
                            neutral = 0
                        elif sentiment == 'negative':
                            negative = 1
                            neutral = 0

                features[f'{char_clean}_mentioned'] = mentioned
                features[f'{char_clean}_positive'] = positive
                features[f'{char_clean}_negative'] = negative
                features[f'{char_clean}_neutral'] = neutral

            return features

        except Exception as e:
            self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
            self.failed_samples.append(sample_id)
            return self.get_empty_characteristic_features()

    def get_empty_characteristic_features(self):
        features = {}
        for char in self.characteristics:
            char_clean = char.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            features[f'{char_clean}_mentioned'] = 0
            features[f'{char_clean}_positive'] = 0
            features[f'{char_clean}_negative'] = 0
            features[f'{char_clean}_neutral'] = 1
        return features


class NLPFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.positive_words = {
            'like', 'likes', 'love', 'loves', 'enjoy', 'enjoys', 'good', 'great',
            'awesome', 'amazing', 'wonderful', 'fantastic', 'excellent', 'perfect',
            'happy', 'fun', 'cool', 'nice', 'best', 'favorite', 'prefer', 'prefers'
        }
        self.negative_words = {
            'dislike', 'dislikes', 'hate', 'hates', 'bad', 'terrible', 'awful',
            'horrible', 'worst', 'not', 'never', 'no', 'dont', "don't", 'doesnt',
            "doesn't", 'cannot', "can't", 'wont', "won't"
        }

    def extract_nlp_features(self, text):
        if pd.isna(text) or text == "":
            return self.get_empty_nlp_features()

        try:
            text = str(text).lower()
            words = word_tokenize(text)
            sentences = sent_tokenize(text)

            features = {}

            features['word_count'] = len(words)
            features['sentence_count'] = len(sentences)
            features['char_count'] = len(text)
            features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
            features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0

            features['shortness_score'] = 1 / (1 + features['word_count'])

            unique_words = set(words)
            features['lexical_diversity'] = len(unique_words) / len(words) if words else 0

            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity

            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)

            features['positive_word_count'] = positive_count
            features['negative_word_count'] = negative_count
            features['positive_word_ratio'] = positive_count / len(words) if words else 0
            features['negative_word_ratio'] = negative_count / len(words) if words else 0

            try:
                features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            except:
                features['flesch_reading_ease'] = 0
                features['flesch_kincaid_grade'] = 0

            return features

        except Exception as e:
            return self.get_empty_nlp_features()

    def get_empty_nlp_features(self):
        return {
            'word_count': 0, 'sentence_count': 0, 'char_count': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0, 'shortness_score': 1,
            'lexical_diversity': 0, 'sentiment_polarity': 0, 'sentiment_subjectivity': 0,
            'positive_word_count': 0, 'negative_word_count': 0,
            'positive_word_ratio': 0, 'negative_word_ratio': 0,
            'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0
        }


def process_features(df):
    processed_df = df.copy()

    scaler = StandardScaler()
    # processed_df['FSR_scaled'] = scaler.fit_transform(processed_df[['FSR']])
    # processed_df['avg_PE_scaled'] = scaler.fit_transform(processed_df[['avg_PE']])

    # char_extractor = CharacteristicFeatureExtractor()
    nlp_extractor = NLPFeatureExtractor()

    # print("Extracting characteristic features...")
    # for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df), desc="Characteristic features"):
    #     char_features = char_extractor.extract_characteristic_features(row['free_response'], idx)
    #     for feature_name, feature_value in char_features.items():
    #         processed_df.loc[idx, feature_name] = feature_value

    print("Extracting NLP features...")
    for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df), desc="NLP features"):
        nlp_features = nlp_extractor.extract_nlp_features(row['free_response_TDprof_norm'])
        for feature_name, feature_value in nlp_features.items():
            processed_df.loc[idx, feature_name] = feature_value

    return processed_df


def preprocess_prediction_data(df, is_test_data=False):
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    if is_test_data:
        test_preprocessed_path = project_root / "data" / data_version / f"LLM_data_test_preprocessed_{model_version}.csv"

        if test_preprocessed_path.exists():
            print(f"Loading preprocessed test data from: {test_preprocessed_path}")
            return pd.read_csv(test_preprocessed_path)
        else:
            print("Preprocessed test data not found. Processing test data...")

    artifacts_dir = project_root / "Results" / model_version / "preprocessing"
    info_path = artifacts_dir / 'preprocessing_info.json'

    if not info_path.exists():
        raise FileNotFoundError(f"Preprocessing info not found at {info_path}. Please run training first.")

    with open(info_path, 'r') as f:
        preprocessing_info = json.load(f)

    print("Preprocessing info loaded successfully.")

    selected_columns = ['sub', 'FSR', 'TDNorm_avg_PE', 'overall_avg_PE', 'TDnorm_concept_learning',
                        'overall_concept_learning', 'free_response_TDprof_norm', 'td_or_asd', ]
    df_filtered = df[selected_columns].copy()
    df_filtered = df_filtered.dropna()

    processed_df = process_features(df_filtered)

    print(f"Preprocessing completed")
    print(f"Original columns: {len(df.columns)}")
    print(f"Final columns: {len(processed_df.columns)}")
    print(f"Total added features: {len(processed_df.columns) - len(df.columns)}")

    return processed_df


if __name__ == "__main__":
    processed_data = preprocess_training_data()