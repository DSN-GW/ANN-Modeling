import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Agent'))

import pandas as pd
import json
import numpy as np
import re
from collections import Counter

class FeatureExtractor:
    def __init__(self):
        self.characteristics = self.load_characteristics()
        self.agent = None
        self.use_agent = self.initialize_agent()
        # Always set up fallback keywords regardless of agent availability
        # This ensures fallback works even if agent fails during processing
        self.setup_fallback_keywords()
        if not self.use_agent:
            print("Warning: Sonnet agent not available. Using fallback feature extraction.")
    
    def load_characteristics(self):
        char_path = os.path.join('..', '..', 'data', 'Data_v1', 'charactristic.txt')
        with open(char_path, 'r') as f:
            characteristics = [line.strip() for line in f.readlines() if line.strip()]
        return characteristics
    
    def initialize_agent(self):
        try:
            from sonnet_agent import SonnetAgent
            self.agent = SonnetAgent()
            self.setup_agent()
            return True
        except ImportError as e:
            print(f"Cannot import sonnet_agent: {e}")
            print("Please install required dependencies: pip install boto3 python-dotenv")
            return False
        except Exception as e:
            print(f"Error initializing sonnet agent: {e}")
            return False
    
    def setup_fallback_keywords(self):
        self.characteristic_keywords = {
            'personality inference': ['personality', 'character', 'trait', 'behavior', 'like', 'dislike', 'prefer'],
            'sweets': ['sweet', 'candy', 'chocolate', 'cake', 'cookie', 'dessert', 'sugar', 'ice cream'],
            'Fruits and vegetables': ['fruit', 'vegetable', 'apple', 'banana', 'orange', 'carrot', 'broccoli', 'salad'],
            'healthy savory food': ['healthy', 'salad', 'vegetable', 'grain', 'protein', 'nutritious'],
            'food': ['food', 'eat', 'meal', 'lunch', 'dinner', 'breakfast', 'snack', 'hungry'],
            'cosmetics': ['makeup', 'cosmetic', 'lipstick', 'foundation', 'beauty', 'skincare'],
            'fashion': ['clothes', 'fashion', 'shirt', 'dress', 'style', 'outfit', 'wear'],
            'toys, gadgets and games': ['toy', 'game', 'gadget', 'play', 'fun', 'electronic', 'video game'],
            'sports': ['sport', 'basketball', 'football', 'soccer', 'tennis', 'exercise', 'athletic'],
            'music': ['music', 'song', 'guitar', 'piano', 'instrument', 'band', 'listen'],
            'arts and crafts': ['art', 'craft', 'draw', 'paint', 'creative', 'design', 'make']
        }
    
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

Be precise and only mark characteristics as mentioned if there is clear evidence in the text."""
        
        self.agent.set_system_prompt(system_prompt)
        self.agent.set_parameters(max_tokens=2000, temperature=0.1)
    
    def extract_features_from_text(self, text):
        if pd.isna(text) or text == "":
            return self.get_empty_features()
        
        if self.use_agent and self.agent:
            return self.extract_features_with_agent(text)
        else:
            return self.extract_features_with_fallback(text)
    
    def extract_features_with_agent(self, text):
        prompt = f"Analyze this text and extract features for the given characteristics: '{text}'"
        
        try:
            response = self.agent.ask(prompt)
            features = json.loads(response)
            return self.process_features(features)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for text: '{text[:50]}...': {e}")
            return self.extract_features_with_fallback(text)
        except Exception as e:
            print(f"Agent error for text: '{text[:50]}...': {e}")
            return self.extract_features_with_fallback(text)
    
    def extract_features_with_fallback(self, text):
        text_lower = text.lower()
        features = {}
        
        positive_words = ['like', 'likes', 'love', 'loves', 'enjoy', 'enjoys', 'good', 'great', 'awesome']
        negative_words = ['dislike', 'dislikes', 'hate', 'hates', 'not', 'dont', "don't", 'bad', 'terrible']
        
        for char in self.characteristics:
            keywords = self.characteristic_keywords.get(char, [])
            
            mentioned = 0
            sentiment = 0
            
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            
            if keyword_matches > 0:
                mentioned = 1
                
                char_context = []
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        start_idx = text_lower.find(keyword.lower())
                        context_start = max(0, start_idx - 20)
                        context_end = min(len(text), start_idx + len(keyword) + 20)
                        char_context.append(text[context_start:context_end].lower())
                
                context_text = ' '.join(char_context)
                
                positive_count = sum(1 for word in positive_words if word in context_text)
                negative_count = sum(1 for word in negative_words if word in context_text)
                
                if positive_count > negative_count:
                    sentiment = 1
                elif negative_count > positive_count:
                    sentiment = -1
                else:
                    sentiment = 0
            
            features[f"{char}_mentioned"] = mentioned
            features[f"{char}_sentiment"] = sentiment
        
        return features
    
    def get_empty_features(self):
        features = {}
        for char in self.characteristics:
            features[f"{char}_mentioned"] = 0
            features[f"{char}_sentiment"] = 0
        return features
    
    def process_features(self, raw_features):
        processed = {}
        
        for char in self.characteristics:
            char_data = raw_features.get(char, {})
            
            processed[f"{char}_mentioned"] = 1 if char_data.get("mentioned", False) else 0
            
            sentiment = char_data.get("sentiment", "neutral")
            if sentiment == "positive":
                processed[f"{char}_sentiment"] = 1
            elif sentiment == "negative":
                processed[f"{char}_sentiment"] = -1
            else:
                processed[f"{char}_sentiment"] = 0
        
        return processed
    
    def process_dataset(self, df):
        feature_list = []

        for idx, row in df.iterrows():
            print(f"Processing row {idx+1}/{len(df)}")
            features = self.extract_features_from_text(row['free_response'])
            feature_list.append(features)
        
        feature_df = pd.DataFrame(feature_list)
        result_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        
        return result_df