
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

import nltk
from textblob import TextBlob
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------------------
# NLTK setup
# -------------------------------------------------------------------
def _nltk_setup():
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

_nltk_setup()


# ================================================================
# 0. NLP feature extractor - From data_preprocessor
# ================================================================
class NLPFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        self.positive_words = {
            "like","likes","love","loves","enjoy","enjoys","good","great",
            "awesome","amazing","wonderful","fantastic","excellent","perfect",
            "happy","fun","cool","nice","best","favorite","prefer","prefers"
        }
        self.negative_words = {
            "dislike","dislikes","hate","hates","bad","terrible","awful",
            "horrible","worst","not","never","no","dont","don't","doesnt",
            "doesn't","cannot","can't","wont","won't"
        }

    def get_empty_nlp_features(self):
        return {
            "word_count": 0,
            "sentence_count": 0,
            "char_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "shortness_score": 1,
            "lexical_diversity": 0,
            "sentiment_polarity": 0,
            "sentiment_subjectivity": 0,
            "positive_word_count": 0,
            "negative_word_count": 0,
            "positive_word_ratio": 0,
            "negative_word_ratio": 0,
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
        }

    def extract_nlp_features(self, text):
        if pd.isna(text) or str(text).strip() == "":
            return self.get_empty_nlp_features()

        try:
            text = str(text).lower()
            words = word_tokenize(text)
            sentences = sent_tokenize(text)

            feats = {}
            feats["word_count"] = len(words)
            feats["sentence_count"] = len(sentences)
            feats["char_count"] = len(text)
            feats["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
            feats["avg_sentence_length"] = (len(words) / len(sentences)) if sentences else 0

            feats["shortness_score"] = 1 / (1 + feats["word_count"])

            unique_words = set(words)
            feats["lexical_diversity"] = (len(unique_words) / len(words)) if words else 0

            blob = TextBlob(text)
            feats["sentiment_polarity"] = blob.sentiment.polarity
            feats["sentiment_subjectivity"] = blob.sentiment.subjectivity

            pos_ct = sum(w in self.positive_words for w in words)
            neg_ct = sum(w in self.negative_words for w in words)
            feats["positive_word_count"] = pos_ct
            feats["negative_word_count"] = neg_ct
            feats["positive_word_ratio"] = pos_ct / len(words) if words else 0
            feats["negative_word_ratio"] = neg_ct / len(words) if words else 0

            try:
                feats["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
                feats["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            except Exception:
                feats["flesch_reading_ease"] = 0
                feats["flesch_kincaid_grade"] = 0

            return feats
        except Exception:
            return self.get_empty_nlp_features()

def drop_high_corr_features(corr_matrix, threshold=0.85):
    """
    Drops one feature from every pair of highly correlated features.
    """
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    return list(set(to_drop))


def compute_vif(df):
    # Standardize
    X = StandardScaler().fit_transform(df)
    vif = [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]
    return pd.DataFrame({
        "feature": df.columns,
        "VIF": vif
    }).sort_values("VIF", ascending=False)

def iterative_vif_drop(df, threshold=10.0):
    df_current = df.copy()
    removed_features = []

    while True:
        vif_df = compute_vif(df_current)
        max_vif = vif_df["VIF"].max()
        worst_feature = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]

        if max_vif < threshold:
            break

        print(f"Dropping {worst_feature} (VIF={max_vif:.2f})")
        removed_features.append(worst_feature)
        df_current = df_current.drop(columns=[worst_feature])

    return df_current, removed_features

def fetch_final_features(data,text_col,thresh,plot_path,Use_nlp=True):
    df = data.copy()

    if Use_nlp:
        # -------- Extract NLP features --------
        nlp_extractor = NLPFeatureExtractor()

        print(f"Extracting NLP features from `{text_col}`...")
        nlp_features = [nlp_extractor.extract_nlp_features(t) for t in df[text_col]]
        nlp_df = pd.DataFrame(nlp_features)
        df = pd.concat([df.reset_index(drop=True), nlp_df.reset_index(drop=True)], axis=1)

    df.drop(columns=['free_response_ASDprof_norm', 'free_response_ASDprof_unif'], inplace=True)
    df.dropna(inplace=True)

    exclude_cols = ["sub","td_or_asd"]
    df_ = df[exclude_cols]
    num_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != 'object']
    df_num = df[num_cols]

    corr = df_num.corr(method="pearson")

    plt.figure(figsize=(15, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    high_corr_features = drop_high_corr_features(corr, threshold=thresh)
    print(f"\nHighly correlated features {thresh}:")
    print(high_corr_features)

    df_corr_reduced = df_num.drop(columns=high_corr_features, errors='ignore')
    # df_corr_reduced = df_num
    print("Remaining features after correlation filter:", df_corr_reduced.shape[1])

    df_vif_reduced, vif_removed = iterative_vif_drop(df_corr_reduced, threshold=5.0)

    print("\nDropped due to VIF:")
    print(vif_removed)
    print("Final feature count:", df_vif_reduced.shape[1])
    vif_df_final = compute_vif(df_vif_reduced)
    print(vif_df_final)

    # ============================================================
    # FINAL SELECTED FEATURES
    # ============================================================
    selected_features = df_vif_reduced.columns.tolist()
    print("\n=== FINAL SELECTED FEATURES FOR CLUSTERING ===")
    print(selected_features)

    return corr, thresh, high_corr_features, vif_df_final, selected_features,pd.concat([df_,df_num[selected_features]],axis=1)

if __name__ == "__main__":

    # ---------------- USER CONFIG ----------------
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    data_path = data_dir / "Data_Clustering" / "LLM data_aggregate.csv"  # Change here if new file needs to be tested
    TEXT_COLUMN = "free_response_TDprof_norm"  # Can switch if we want to use a different free response

    df = pd.read_csv(data_path)
    print(f"Loaded raw data: {df.shape} from {data_path}")

    _,_,_,_,_,_ = fetch_final_features(df,TEXT_COLUMN,0.3,"",True)