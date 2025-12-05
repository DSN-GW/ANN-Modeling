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
# CONFIG (GENERAL)
# -------------------------------------------------------------------
# Folder:   data/Data_Clustering/<RAW_CSV_NAME>
DATA_VERSION = "Data_Clustering"
RAW_CSV_NAME = "LLM data_aggregate_8.25.25 data_updated 10.8.25.csv"

# Output preprocessed file (shared by all clustering models)
PREPROCESSED_CSV_NAME = "data_preprocessed_clustering.csv"

# 7 numeric columns you use everywhere
NUMERIC_COLUMNS = [
    'FSR', 'BIS', 'SRS.Raw',
    'TDNorm_avg_PE', 'overall_avg_PE',
    'TDnorm_concept_learning', 'overall_concept_learning'
]

# Text column to derive NLP features from
TEXT_COLUMN = 'free_response_TDprof_norm'  # adjust if your column name changes


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
        """Compute all NLP features for a single free-response string."""
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
            features['avg_word_length'] = (
                np.mean([len(word) for word in words]) if words else 0
            )
            features['avg_sentence_length'] = (
                len(words) / len(sentences) if sentences else 0
            )

            # Shortness / brevity heuristic
            features['shortness_score'] = 1 / (1 + features['word_count'])

            # Lexical diversity
            unique_words = set(words)
            features['lexical_diversity'] = (
                len(unique_words) / len(words) if words else 0
            )

            # Sentiment via TextBlob
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity

            # Simple lexicon-based counts
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)

            features['positive_word_count'] = positive_count
            features['negative_word_count'] = negative_count
            features['positive_word_ratio'] = (
                positive_count / len(words) if words else 0
            )
            features['negative_word_ratio'] = (
                negative_count / len(words) if words else 0
            )

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
        """Return a zeroed-out NLP feature dict (for empty/error cases)."""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'shortness_score': 1,
            'lexical_diversity': 0,
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0,
            'positive_word_count': 0,
            'negative_word_count': 0,
            'positive_word_ratio': 0,
            'negative_word_ratio': 0,
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
        }


# -------------------------------------------------------------------
# MAIN PREPROCESSING FUNCTION (GENERAL)
# -------------------------------------------------------------------
def preprocess_clustering_data(
    project_root: Path,
    subset_mode: str = "full",   # "full", "fsr_overlap", "fsr_nonoverlap"
) -> pd.DataFrame:
    """
    General preprocessing for ALL clustering models.

    Steps:
    - Load raw CSV from: data/Data_Clustering/<RAW_CSV_NAME>
    - Drop NaNs in numeric + text + td_or_asd
    - Optionally subset by FSR overlap / non-overlap region
    - Keep original 7 numeric columns
    - Add 7 scaled numeric columns
    - Keep td_or_asd
    - Add NLP feature columns from TEXT_COLUMN
    - Save preprocessed CSV shared by all clustering models:
          data/Data_Clustering/data_preprocessed_clustering.csv
    - Save preprocessing_info.json under Results/Clustering/preprocessing/
    """
    if subset_mode not in ("full", "fsr_overlap", "fsr_nonoverlap"):
        raise ValueError(f"Invalid subset_mode='{subset_mode}'. "
                         f"Use 'full', 'fsr_overlap', or 'fsr_nonoverlap'.")

    # ----------------------------------------------------------------
    # 1. Load raw data
    # ----------------------------------------------------------------
    data_path = (
        project_root
        / "data"
        / DATA_VERSION
        / RAW_CSV_NAME
    )
    df = pd.read_csv(data_path)
    print(f"Loaded raw data: {df.shape} from {data_path}")

    # ----------------------------------------------------------------
    # 2. Check required columns
    # ----------------------------------------------------------------
    required_cols = NUMERIC_COLUMNS + [TEXT_COLUMN, 'td_or_asd']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # ----------------------------------------------------------------
    # 3. Drop NaNs on numeric + text + target
    # ----------------------------------------------------------------
    df_filtered = df.dropna(
        subset=NUMERIC_COLUMNS + [TEXT_COLUMN, 'td_or_asd']
    ).reset_index(drop=True)
    print(f"After dropping NaNs: {df_filtered.shape}")

    # ----------------------------------------------------------------
    # 3a. Optional FSR overlap / non-overlap subset
    # ----------------------------------------------------------------
    df_subset = df_filtered.copy()
    overlap_info = None

    if subset_mode in ("fsr_overlap", "fsr_nonoverlap"):
        # Compute class-wise FSR ranges
        class0 = df_filtered[df_filtered['td_or_asd'] == 0]['FSR']
        class1 = df_filtered[df_filtered['td_or_asd'] == 1]['FSR']

        if class0.empty or class1.empty:
            print("⚠️ One of the classes has no rows after filtering; "
                  "falling back to FULL dataset for clustering.")
        else:
            min0, max0 = class0.min(), class0.max()
            min1, max1 = class1.min(), class1.max()

            overlap_min = max(min0, min1)
            overlap_max = min(max0, max1)

            if overlap_min <= overlap_max:
                mask_overlap = (
                    (df_filtered['FSR'] >= overlap_min) &
                    (df_filtered['FSR'] <= overlap_max)
                )

                if subset_mode == "fsr_overlap":
                    df_subset = df_filtered[mask_overlap].reset_index(drop=True)
                    print(f"[FSR Overlap] Range: [{overlap_min:.4f}, {overlap_max:.4f}] "
                          f"→ {df_subset.shape[0]} rows")
                elif subset_mode == "fsr_nonoverlap":
                    df_subset = df_filtered[~mask_overlap].reset_index(drop=True)
                    print(f"[FSR Non-overlap] Removing range "
                          f"[{overlap_min:.4f}, {overlap_max:.4f}] "
                          f"→ {df_subset.shape[0]} rows")

                overlap_info = {
                    "subset_mode": subset_mode,
                    "overlap_min": float(overlap_min),
                    "overlap_max": float(overlap_max),
                    "subset_shape": list(df_subset.shape),
                }
            else:
                print("⚠️ No FSR overlap between class 0 and class 1; "
                      "falling back to FULL dataset for clustering.")
                subset_mode = "full"

    if subset_mode == "full":
        print("Using FULL participant set for clustering (no FSR subsetting).")

    # At this point, df_subset is what we actually preprocess
    print(f"Subset used for preprocessing: {df_subset.shape} (mode='{subset_mode}')")

    # ----------------------------------------------------------------
    # 4. Scale numeric features (on chosen subset)
    # ----------------------------------------------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_subset[NUMERIC_COLUMNS])
    scaled_cols = [f"{c}_scaled" for c in NUMERIC_COLUMNS]
    scaled_df = pd.DataFrame(scaled_data, columns=scaled_cols)

    # ----------------------------------------------------------------
    # 5. NLP feature extraction from text column (on chosen subset)
    # ----------------------------------------------------------------
    nlp_extractor = NLPFeatureExtractor()
    nlp_features_list = []

    print(f"Extracting NLP features from `{TEXT_COLUMN}` on subset...")
    for text in df_subset[TEXT_COLUMN]:
        feats = nlp_extractor.extract_nlp_features(text)
        nlp_features_list.append(feats)

    nlp_df = pd.DataFrame(nlp_features_list)

    # ----------------------------------------------------------------
    # 6. Build final DataFrame
    # ----------------------------------------------------------------
    # Order: 7 original numeric → 7 scaled numeric → td_or_asd → NLP features
    df_final = pd.concat(
        [
            df_subset[NUMERIC_COLUMNS].reset_index(drop=True),
            scaled_df.reset_index(drop=True),
            df_subset[['td_or_asd']].reset_index(drop=True),
            nlp_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # ----------------------------------------------------------------
    # 7. Save preprocessed data (shared by all clustering runs)
    # ----------------------------------------------------------------
    preprocessed_path = (
        project_root
        / "data"
        / DATA_VERSION
        / PREPROCESSED_CSV_NAME
    )
    df_final.to_csv(preprocessed_path, index=False)
    print(f"Saved preprocessed clustering data → {preprocessed_path}")

    # ----------------------------------------------------------------
    # 8. Save preprocessing metadata
    # ----------------------------------------------------------------
    results_dir = project_root / "Results" / "Clustering" / "preprocessing"
    results_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "raw_path": str(data_path),
        "preprocessed_path": str(preprocessed_path),
        "original_shape": list(df.shape),
        "filtered_shape_after_nan_drop": list(df_filtered.shape),
        "subset_mode": subset_mode,
        "subset_shape": list(df_subset.shape),
        "dropped_rows_total": int(df.shape[0] - df_subset.shape[0]),
        "numeric_original_columns": NUMERIC_COLUMNS,
        "scaled_columns": scaled_cols,
        "text_column": TEXT_COLUMN,
        "nlp_feature_columns": list(nlp_df.columns),
        "final_columns": list(df_final.columns),
        "fsr_overlap_info": overlap_info,
    }

    with open(results_dir / "preprocessing_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved preprocessing metadata → {results_dir / 'preprocessing_info.json'}")

    return df_final


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print("\n=== Running General Clustering Preprocessing ===\n")
    # Default run: full participants
    preprocess_clustering_data(project_root, subset_mode="full")
