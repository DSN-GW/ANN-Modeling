import sys
import os
import pandas as pd
import numpy as np
import json
import boto3
import time
import logging
from datetime import datetime
from pathlib import Path
from botocore.exceptions import ClientError
import re

# Add the Agent directory to the path
current_dir = Path(__file__).parent
agent_dir = current_dir.parent / "Agent"
sys.path.append(str(agent_dir))

from sonnet_agent import SonnetAgent

class CharacteristicFeatureExtractor:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.agent = SonnetAgent()
        
        # Get the project root directory (two levels up from current file)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        # Setup logging
        log_dir = project_root / "Results" / "V1" / "logs"
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
        
        # Load characteristics
        char_path = project_root / "data" / "Data_v1" / "charactristic.txt"
        self.characteristics = self.load_characteristics(char_path)
        
        # Initialize feature storage
        self.feature_data = {}
        # Always ensure failed_samples is initialized
        if not hasattr(self, 'failed_samples'):
            self.failed_samples = []
    
    def load_characteristics(self, char_path):
        """Load characteristics from file."""
        try:
            with open(char_path, 'r', encoding='utf-8') as f:
                characteristics = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(characteristics)} characteristics")
            return characteristics
        except FileNotFoundError:
            print(f"Warning: Characteristics file not found at {char_path}")
            return []
    
    def initialize_agent(self):
        try:
            from sonnet_agent import SonnetAgent
            self.agent = SonnetAgent()
            self.setup_agent()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {str(e)}")
            return False

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
        self.agent.set_parameters(max_tokens=4000, temperature=0.1)  # Increased max_tokens for batch processing
    
    def create_batch_prompt(self, texts_with_indices):
        prompt = "Analyze the following texts and extract features for the given characteristics. Return a JSON array with results for each text:\n\n"
        
        for idx, text in texts_with_indices:
            prompt += f"Text {idx+1}: '{text}'\n"
        
        prompt += "\nReturn your analysis as a JSON array with the following structure:\n"
        prompt += "[\n"
        prompt += "  {\n"
        prompt += "    \"text_index\": 1,\n"
        prompt += "    \"features\": {\n"
        for char in self.characteristics:
            prompt += f"      \"{char}\": {{\n"
            prompt += "        \"mentioned\": true/false,\n"
            prompt += "        \"sentiment\": \"positive/negative/neutral\"\n"
            prompt += "      },\n"
        prompt += "    }\n"
        prompt += "  }\n"
        prompt += "]\n\n"
        prompt += "IMPORTANT: Return exactly one result object per text, maintaining the same order."
        return prompt
    
    def extract_features_from_text(self, text):
        if pd.isna(text) or text == "":
            return self.get_empty_features()
        try:
            return self.extract_features_with_agent(text)
        except Exception as e:
            self.logger.error(f"Failed to extract features from text: {str(e)}")
            self.logger.error(f"Text content: {text[:200]}...")  # Log first 200 chars
            return self.get_empty_features()

    def extract_json_from_response(self, response):
        import re
        if not response or response.strip() == "":
            raise ValueError("Empty response from agent")
        
        json_pattern = r'\{.*\}'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            return json_str
        else:
            return response
    
    def extract_features_with_agent(self, text):
        try:
            prompt = f"Analyze this text and extract features for the given characteristics: '{text}'"
            response = self.agent.ask(prompt)
            if not response:
                raise ValueError("Empty response from agent")
            json_response = self.extract_json_from_response(response)
            if not json_response.strip():
                raise ValueError("Empty JSON response")
            
            features = json.loads(json_response)
            return self.process_features(features)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            self.logger.error(f"Raw response: {response}")
            raise
        except Exception as e:
            self.logger.error(f"Error in extract_features_with_agent: {str(e)}")
            raise

    def extract_features_batch(self, texts):
        try:
            batch_prompt = self.create_batch_prompt([(i, text) for i, text in enumerate(texts)])
            
            response = self.agent.ask(batch_prompt)
            
            if not response:
                raise ValueError("Empty response from agent")
            
            try:
                batch_results = json.loads(response)
                if isinstance(batch_results, list):
                    return batch_results
            except json.JSONDecodeError:
                pass
            
            import re
            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            
            if len(json_objects) == len(texts):
                batch_results = []
                for i, json_str in enumerate(json_objects):
                    try:
                        features = json.loads(json_str)
                        batch_results.append({
                            "text_index": i + 1,
                            "features": features
                        })
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse JSON for text {i+1}")
                        batch_results.append({
                            "text_index": i + 1,
                            "features": self.get_empty_features_dict()
                        })
                return batch_results
            else:
                raise ValueError(f"Expected {len(texts)} results, got {len(json_objects)}")
                
        except Exception as e:
            self.logger.error(f"Error in extract_features_batch: {str(e)}")
            return [{"text_index": i+1, "features": self.get_empty_features_dict()}
                   for i in range(len(texts))]

    def get_empty_features_dict(self):
        features = {}
        for char in self.characteristics:
            features[char] = {
                "mentioned": False,
                "sentiment": "neutral"
            }
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
    
    def save_failed_samples(self):
        # Ensure failed_samples exists
        if not hasattr(self, 'failed_samples'):
            self.failed_samples = []
            
        if self.failed_samples:
            failed_df = pd.DataFrame(self.failed_samples)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get the project root directory (two levels up from current file)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            failed_file = project_root / "Results" / "V1" / "logs" / f'failed_samples_{timestamp}.csv'
            
            failed_df.to_csv(failed_file, index=False)
            self.logger.info(f"Saved {len(self.failed_samples)} failed samples to {failed_file}")
    
    def process_dataset(self, df):
        # Ensure failed_samples exists
        if not hasattr(self, 'failed_samples'):
            self.failed_samples = []
            
        feature_list = []
        df = df.reset_index(drop=True)
        total_samples = len(df)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        print(f"Processing {total_samples} samples in {num_batches} batches of size {self.batch_size}")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_samples)
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx + 1}-{end_idx})")
            batch_texts = []
            for idx in range(start_idx, end_idx):
                text = df.iloc[idx]['free_response']
                batch_texts.append((idx, text))
            
            try:
                # Extract just the texts for the batch processing
                texts_only = [text for idx, text in batch_texts]
                batch_results = self.extract_features_batch(texts_only)
                
                for i, result in enumerate(batch_results):
                    text_index = result["text_index"]
                    actual_idx = start_idx + text_index - 1
                    
                    if actual_idx < len(df):
                        try:
                            features = self.process_features(result["features"])
                            feature_list.append(features)
                        except Exception as e:
                            self.logger.error(f"Failed to process features for sample {actual_idx + 1}: {str(e)}")
                            feature_list.append(self.get_empty_features())
                    else:
                        self.logger.error(f"Invalid text index: {text_index}")
                        feature_list.append(self.get_empty_features())
                
            except Exception as e:
                self.logger.error(f"Failed to process batch {batch_idx + 1}: {str(e)}")
                for idx in range(start_idx, end_idx):
                    feature_list.append(self.get_empty_features())
                    failed_sample = {
                        'row_index': idx,
                        'sub': df.iloc[idx].get('sub', 'unknown'),
                        'text_preview': str(df.iloc[idx].get('free_response', ''))[:200],
                        'error_message': f"Batch processing failed: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    }
                    self.failed_samples.append(failed_sample)
        
        # Check failed_samples again before using it
        if hasattr(self, 'failed_samples') and self.failed_samples:
            self.save_failed_samples()
            self.logger.warning(f"Total failed samples: {len(self.failed_samples)} out of {len(df)}")
            
        feature_df = pd.DataFrame(feature_list)
        result_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        return result_df