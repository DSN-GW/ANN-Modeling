import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class AutismClassifier(nn.Module):
    def __init__(self, num_numerical_features, device, transformer_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        super(AutismClassifier, self).__init__()
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer = AutoModel.from_pretrained(transformer_model)
        
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
        for layer in self.transformer.encoder.layer[:6]:  # Freeze first 6 layers
            for param in layer.parameters():
                param.requires_grad = False
        
        transformer_output_dim = self.transformer.config.hidden_size
        
        self.text_projection = nn.Sequential(
            nn.Linear(transformer_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Enhanced numerical processing with deeper network
        self.numerical_projection = nn.Sequential(
            nn.Linear(num_numerical_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=96, num_heads=4, batch_first=True)
        
        combined_dim = 64 + 32
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, texts, numerical_features):
        # Process text data
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        transformer_output = self.transformer(**encoded_input)
        cls_output = transformer_output.last_hidden_state[:, 0, :]
        
        text_features = self.text_projection(cls_output)
        numerical_features_processed = self.numerical_projection(numerical_features)
        
        combined_features = torch.cat([text_features, numerical_features_processed], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        if combined_features.size(1) == fused_features.size(1):
            fused_features = fused_features + combined_features
        
        logits = self.classifier(fused_features)
        
        return logits