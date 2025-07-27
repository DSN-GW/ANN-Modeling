import pandas as pd
import numpy as np
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
from textblob import TextBlob
import nltk
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class NLPFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
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
    
    def extract_basic_features(self, text):
        if pd.isna(text) or text == "":
            return self.get_empty_basic_features()
        
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
        
        return features
    
    def extract_sentiment_features(self, text):
        if pd.isna(text) or text == "":
            return self.get_empty_sentiment_features()
        
        text = str(text).lower()
        words = word_tokenize(text)
        
        features = {}
        
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        features['positive_word_count'] = positive_count
        features['negative_word_count'] = negative_count
        features['positive_word_ratio'] = positive_count / len(words) if words else 0
        features['negative_word_ratio'] = negative_count / len(words) if words else 0
        
        features['positive_attributes'] = positive_count > negative_count
        
        return features
    
    def extract_cohesiveness_features(self, text):
        if pd.isna(text) or text == "":
            return self.get_empty_cohesiveness_features()
        
        text = str(text)
        sentences = sent_tokenize(text)
        
        features = {}
        
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
        
        connector_words = {'and', 'but', 'or', 'so', 'because', 'since', 'although', 'however', 'therefore', 'moreover'}
        words = word_tokenize(text.lower())
        connector_count = sum(1 for word in words if word in connector_words)
        features['connector_ratio'] = connector_count / len(words) if words else 0
        
        if len(sentences) > 1:
            sentence_similarities = []
            for i in range(len(sentences) - 1):
                blob1 = TextBlob(sentences[i])
                blob2 = TextBlob(sentences[i + 1])
                
                words1 = set(blob1.words.lower())
                words2 = set(blob2.words.lower())
                
                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    sentence_similarities.append(similarity)
            
            features['cohesiveness_score'] = np.mean(sentence_similarities) if sentence_similarities else 0
        else:
            features['cohesiveness_score'] = 1.0 if sentences else 0
        
        return features
    
    def extract_linguistic_features(self, text):
        if pd.isna(text) or text == "":
            return self.get_empty_linguistic_features()
        
        text = str(text)
        words = word_tokenize(text)
        
        features = {}
        
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        total_words = len(words)
        features['noun_ratio'] = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                                pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_words if total_words else 0
        features['verb_ratio'] = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                                pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                                pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_words if total_words else 0
        features['adj_ratio'] = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                               pos_counts.get('JJS', 0)) / total_words if total_words else 0
        features['adv_ratio'] = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                               pos_counts.get('RBS', 0)) / total_words if total_words else 0
        
        punctuation_count = len(re.findall(r'[.!?;,:]', text))
        features['punctuation_ratio'] = punctuation_count / len(text) if text else 0
        
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        return features
    
    def get_empty_basic_features(self):
        return {
            'word_count': 0, 'sentence_count': 0, 'char_count': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0, 'shortness_score': 1,
            'lexical_diversity': 0
        }
    
    def get_empty_sentiment_features(self):
        return {
            'sentiment_polarity': 0, 'sentiment_subjectivity': 0,
            'positive_word_count': 0, 'negative_word_count': 0,
            'positive_word_ratio': 0, 'negative_word_ratio': 0,
            'positive_attributes': False
        }
    
    def get_empty_cohesiveness_features(self):
        return {
            'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
            'connector_ratio': 0, 'cohesiveness_score': 0
        }
    
    def get_empty_linguistic_features(self):
        return {
            'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0,
            'punctuation_ratio': 0, 'exclamation_count': 0, 'question_count': 0
        }
    
    def extract_all_features(self, text):
        features = {}
        features.update(self.extract_basic_features(text))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_cohesiveness_features(text))
        features.update(self.extract_linguistic_features(text))
        return features
    
    def process_dataset(self, df, text_column='free_response'):
        feature_list = []
        for idx, row in df.iterrows():
            print(f"Processing NLP features for row {idx+1}/{len(df)}")
            features = self.extract_all_features(row[text_column])
            feature_list.append(features)
        
        feature_df = pd.DataFrame(feature_list)
        result_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        
        return result_df