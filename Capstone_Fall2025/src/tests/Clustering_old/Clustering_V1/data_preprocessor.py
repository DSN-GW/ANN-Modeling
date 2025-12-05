# data_preprocessor.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

import nltk
from textblob import TextBlob
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------------------
# NLTK setup
# -------------------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
data_version = "Data_Clustering_v1"
model_version = "Clustering_V1"

NUMERIC_COLUMNS = [
    'FSR', 'BIS', 'SRS.Raw',
    'TDNorm_avg_PE', 'overall_avg_PE',
    'TDnorm_concept_learning', 'overall_concept_learning'
]

TEXT_COLUMN = 'free_response_TDprof_norm'  # adjust here if your text column has a different name


# -------------------------------------------------------------------
# NLP Feature Extractor
# -------------------------------------------------------------------
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

            # Basic counts
            features['word_count'] = len(words)
            features['sentence_count'] = len(sentences)
            features['char_count'] = len(text)
            features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
            features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0

            # Shortness / brevity heuristic
            features['shortness_score'] = 1 / (1 + features['word_count'])

            # Lexical diversity
            unique_words = set(words)
            features['lexical_diversity'] = len(unique_words) / len(words) if words else 0

            # Sentiment via TextBlob
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity

            # Simple lexicon-based counts
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)

            features['positive_word_count'] = positive_count
            features['negative_word_count'] = negative_count
            features['positive_word_ratio'] = positive_count / len(words) if words else 0
            features['negative_word_ratio'] = negative_count / len(words) if words else 0

            # Readability metrics
            try:
                features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            except Exception:
                features['flesch_reading_ease'] = 0
                features['flesch_kincaid_grade'] = 0

            return features

        except Exception:
            # On any error, fall back to zeros
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


# -------------------------------------------------------------------
# MAIN PREPROCESSING FUNCTION
# -------------------------------------------------------------------
def preprocess_clustering_data(project_root: Path) -> pd.DataFrame:
    """

    - Loads raw CSV
    - Drops NaNs in numeric + text column
    - Keeps original 7 numeric columns
    - Adds 7 scaled numeric columns
    - Adds td_or_asd
    - Adds NLP feature columns from free_response
    - Saves preprocessed CSV
    """
    data_path = project_root / "data" / data_version / "LLM data_aggregate_8.25.25 data_updated 10.8.25.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    # Ensure required columns exist
    required_cols = NUMERIC_COLUMNS + [TEXT_COLUMN, 'td_or_asd']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Drop NaNs on numeric + text + target
    df_filtered = df.dropna(subset=NUMERIC_COLUMNS + [TEXT_COLUMN, 'td_or_asd']).reset_index(drop=True)
    print(f"After dropping NaNs: {df_filtered.shape}")

    # ---- Scale numeric features ----
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[NUMERIC_COLUMNS])
    scaled_cols = [f"{c}_scaled" for c in NUMERIC_COLUMNS]
    scaled_df = pd.DataFrame(scaled_data, columns=scaled_cols)

    # ---- NLP feature extraction from free_response ----
    nlp_extractor = NLPFeatureExtractor()
    nlp_features_list = []

    print("Extracting NLP features from free_response...")
    for text in df_filtered[TEXT_COLUMN]:
        feats = nlp_extractor.extract_nlp_features(text)
        nlp_features_list.append(feats)

    nlp_df = pd.DataFrame(nlp_features_list)

    # ---- Build final DataFrame ----
    # Order: 7 original numeric → 7 scaled numeric → td_or_asd → NLP features
    df_final = pd.concat(
        [
            df_filtered[NUMERIC_COLUMNS].reset_index(drop=True),
            scaled_df.reset_index(drop=True),
            df_filtered[['td_or_asd']].reset_index(drop=True),
            nlp_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Save preprocessed data
    preprocessed_path = project_root / "data" / data_version / f"data_preprocessed_{model_version}.csv"
    df_final.to_csv(preprocessed_path, index=False)
    print(f"Saved preprocessed data → {preprocessed_path}")

    # Save preprocessing metadata
    results_dir = project_root / "Results" / model_version / "preprocessing"
    results_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "original_shape": df.shape,
        "filtered_shape": df_filtered.shape,
        "dropped_rows": int(df.shape[0] - df_filtered.shape[0]),
        "numeric_original_columns": NUMERIC_COLUMNS,
        "scaled_columns": scaled_cols,
        "text_column": TEXT_COLUMN,
        "nlp_feature_columns": list(nlp_df.columns),
        "final_columns": list(df_final.columns)
    }

    with open(results_dir / "preprocessing_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved preprocessing metadata → {results_dir / 'preprocessing_info.json'}")

    return df_final


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print(f"\n=== Running {model_version} Preprocessing ===\n")
    preprocess_clustering_data(project_root)
