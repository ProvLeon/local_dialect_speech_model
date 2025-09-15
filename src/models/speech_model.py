#!/usr/bin/env python3
"""
This is the python file for our updated model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..preprocessing.audio_processor import AudioProcessor
import logging
import json


logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out(context)

class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block with depthwise separable convolutions"""
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.depthwise = nn.Conv1d(
            input_channels, input_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size//2 * dilation,
            groups=input_channels, dilation=dilation
        )
        self.pointwise = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(output_channels)

        self.skip = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(output_channels)
        ) if input_channels != output_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.bn2(out)
        identity = self.skip(identity)
        out += identity
        out = self.relu(out)
        return out

class ImprovedTwiSpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_slot_classes, slot_value_maps, num_layers=2, dropout=0.5, num_heads=4):
        super(ImprovedTwiSpeechModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_slot_classes = num_slot_classes
        self.slot_value_maps = slot_value_maps

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=128,  # Output from conv2
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = MultiHeadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.intent_classifier = nn.Linear(hidden_dim, num_classes)
        self.slot_classifier = nn.Linear(hidden_dim, num_slot_classes)

        # Create a ModuleDict for slot value classifiers
        self.slot_value_classifiers = nn.ModuleDict()
        for slot_type, value_map in self.slot_value_maps.items():
            self.slot_value_classifiers[slot_type] = nn.Linear(hidden_dim, len(value_map))

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] != self.input_dim:
            x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        pooled_output = attn_out.mean(dim=1)
        shared_features = self.shared_layer(pooled_output)

        intent_logits = self.intent_classifier(shared_features)
        slot_type_logits = self.slot_classifier(shared_features)

        slot_value_logits = {}
        for slot_type, classifier in self.slot_value_classifiers.items():
            slot_value_logits[slot_type] = classifier(shared_features)

        return intent_logits, slot_type_logits, slot_value_logits

class IntentOnlyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5, num_heads=4):
        super(IntentOnlyModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=128,  # Output from conv2
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = MultiHeadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.intent_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] != self.input_dim:
            x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        pooled_output = attn_out.mean(dim=1)
        shared_features = self.shared_layer(pooled_output)

        intent_logits = self.intent_classifier(shared_features)
        return intent_logits

class EnhancedTwiSpeechModel(nn.Module):
    """
    Advanced speech recognition model with improved architecture
    including squeeze-and-excitation blocks, attention mechanisms, and residual connections.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5, num_heads=4):
        super(EnhancedTwiSpeechModel, self).__init__()

        self.input_dim = input_dim
        print(f"Creating EnhancedTwiSpeechModel with input_dim={input_dim}")

        self.se_block = SqueezeExciteBlock(input_dim)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )
        self.conv_blocks = nn.ModuleList([
            ResidualBlock(
                input_channels=input_dim if i==0 else 64*(2**min(i-1, 2)),
                output_channels=64*(2**min(i, 2)),
                kernel_size=5 if i==0 else 3,
                stride=1,
                dilation=i+1
            )
            for i in range(3)
        ])
        self.temporal_pool = nn.AdaptiveAvgPool1d(50)
        conv_output_dim = 64 * (2**min(2, 2))
        self.dim_matching = nn.Linear(conv_output_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = MultiHeadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        """Improved weight initialization for faster convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 3:
            if x.shape[1] != self.input_dim and x.shape[2] == self.input_dim:
                x = x.transpose(1, 2)
        x = self.se_block(x)
        x = self.initial_conv(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.temporal_pool(x)
        x = x.transpose(1, 2)
        x = self.dim_matching(x)
        gru_out, _ = self.gru(x)
        attn_out = self.attention(gru_out)
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        pooled = avg_pool + max_pool
        features = self.classification_head(pooled)
        output = self.fc_out(features)
        return output

class IntentClassifier:
    def __init__(self, model_path, device, processor=None, label_map_path=None, model_type='standard'):
        self.device = device
        self.processor = processor if processor else AudioProcessor()
        self.model_type = model_type
        if label_map_path and os.path.exists(label_map_path):
            self.label_map = np.load(label_map_path, allow_pickle=True).item()
            num_classes = len(self.label_map)
        else:
            self.label_map = None
            num_classes = 20
        input_dim = self._determine_input_dim(model_path)
        logger.info(f"Using input dimension: {input_dim}")
        if model_type == 'enhanced':
            logger.info("Initializing enhanced speech model...")
            self.model = EnhancedTwiSpeechModel(
                input_dim=input_dim,
                hidden_dim=128,
                num_classes=num_classes
            )
        else:
            logger.info("Initializing standard speech model...")
            self.model = ImprovedTwiSpeechModel(
                input_dim=input_dim,
                hidden_dim=128,
                num_classes=num_classes
            )
        self._load_model_weights(model_path)
        self.model.eval()

    def _determine_input_dim(self, model_path):
        model_dir = os.path.dirname(model_path)
        model_info_path = os.path.join(model_dir, 'model_info.json')
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    return model_info.get('input_dim', 94)
            except Exception as e:
                logger.warning(f"Could not read model_info.json: {e}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                return checkpoint['config'].get('input_dim', 94)
        except Exception as e:
            logger.warning(f"Could not extract input_dim from checkpoint: {e}")
        return 94

    def _load_model_weights(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                logger.warning(f"Unexpected checkpoint format, attempting to load directly")
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise

    def classify(self, audio_path):
        features = self.processor.preprocess(audio_path)
        input_channels = self.model.input_dim if hasattr(self.model, 'input_dim') else 94
        if features.shape[0] != input_channels:
            logger.warning(f"Feature dimension mismatch. Model expects {input_channels}, got {features.shape[0]}")
            if features.shape[0] > input_channels:
                features = features[:input_channels, :]
            else:
                padding = np.zeros((input_channels - features.shape[0], features.shape[1]))
                features = np.vstack((features, padding))
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        if self.label_map:
            idx_to_label = {v: k for k, v in self.label_map.items()}
            intent = idx_to_label.get(predicted_idx.item(), f"unknown_{predicted_idx.item()}")
        else:
            intent = f"intent_{predicted_idx.item()}"
        return intent, confidence.item()
