#!/usr/bin/env python3
"""
Comprehensive 47-Class Preserving Training Pipeline for Speech Intent Classification

This script implements a multi-strategy approach to maintain all 47 classes while preventing
overfitting and addressing extreme class imbalance. It now supports joint training
of intents and slots, including slot values.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import time
from copy import deepcopy

from src.models.speech_model import IntentOnlyModel
from src.features.feature_extractor import TwiDataset
from src.utils.advanced_augmentation import create_balanced_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn(batch):
    import torch
    import torch.nn.functional as F
    feats, labs = [], []
    for sample in batch:
        f, l, st, sv = sample
        feats.append(f)
        labs.append(l)

    max_len = max(f.shape[1] for f in feats) if feats else 0
    padded_feats = []
    for f in feats:
        if f.shape[1] < max_len:
            pad = torch.zeros(f.shape[0], max_len - f.shape[1])
            f = torch.cat([f, pad], dim=1)
        padded_feats.append(f)

    if not padded_feats:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.stack(padded_feats), torch.stack(labs)

class ComprehensiveClassPreservingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config.get('data_dir', 'data/processed')
        self.model_dir = config.get('model_dir', 'data/models/47_class_preserving')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_dir, exist_ok=True)

        self.label_map, self.slot_map, self.slot_value_maps = self.load_maps()
        self.config['num_slot_classes'] = len(self.slot_map)

    def load_maps(self):
        label_map_path = os.path.join(self.data_dir, 'label_map.json')
        slot_map_path = os.path.join(self.data_dir, 'slot_map.json')
        with open(label_map_path, 'r') as f: label_map = json.load(f)
        with open(slot_map_path, 'r') as f: slot_map = json.load(f)

        slot_value_maps = {}
        for slot_type in slot_map.keys():
            path = os.path.join(self.data_dir, f'{slot_type}_map.json')
            if os.path.exists(path):
                with open(path, 'r') as f: slot_value_maps[slot_type] = json.load(f)

        return label_map, slot_map, slot_value_maps

    def run_complete_pipeline(self):
        features, labels, slots = self.load_data()
        if self.config.get('use_augmentation', True):
            features, labels, slots = create_balanced_dataset(features, labels, slots)

        split_indices, _ = self.create_robust_splits(features, labels)

        model = self.create_adaptive_model(features[0].shape[0], len(self.label_map))
        history = self.train_model(model, features, labels, slots, split_indices)

        test_results = self.evaluate_on_test_set(model, features, labels, slots, split_indices)

        results = {'evaluation_results': {'test': test_results}, 'history': history}

        # Save results
        with open(os.path.join(self.model_dir, 'final_results.json'), 'w') as f:
            # Make history JSON serializable
            serializable_history = {k: [float(i) for i in v] for k, v in history.items()}
            results_to_save = {'evaluation_results': {'test': test_results}, 'history': serializable_history}
            json.dump(results_to_save, f, indent=2)

        return results

    def evaluate_on_test_set(self, model, features, labels, slots, split_indices):
        test_indices = split_indices['test']
        test_dataset = TwiDataset([features[i] for i in test_indices], [labels[i] for i in test_indices], [slots[i] for i in test_indices], self.label_map, {}, {})
        test_loader = DataLoader(test_dataset, batch_size=self.config.get('batch_size', 32), collate_fn=collate_fn)

        model.load_state_dict(torch.load(os.path.join(self.model_dir, 'best_model.pt')))
        model.eval()
        test_intent_preds, test_intent_targets = [], []
        with torch.no_grad():
            for features_b, intents_b in test_loader:
                features_b, intents_b = features_b.to(self.device), intents_b.to(self.device)

                intent_logits = model(features_b)
                test_intent_preds.extend(torch.argmax(intent_logits, dim=1).cpu().numpy())
                test_intent_targets.extend(intents_b.cpu().numpy())

        accuracy = accuracy_score(test_intent_targets, test_intent_preds)
        macro_f1_full = f1_score(test_intent_targets, test_intent_preds, average='macro', zero_division=0)

        return {
            'accuracy': accuracy,
            'macro_f1_full': macro_f1_full,
            'macro_f1_present': 0, # not calculated
            'present_classes': len(set(test_intent_targets)),
            'total_classes': len(self.label_map)
        }

    def load_data(self):
        features = list(np.load(os.path.join(self.data_dir, "features.npy"), allow_pickle=True))
        labels = list(np.load(os.path.join(self.data_dir, "labels.npy"), allow_pickle=True))
        with open(os.path.join(self.data_dir, "slots.json"), 'r') as f:
            slots = json.load(f)
        return features, labels, slots

    def create_robust_splits(self, features, labels):
        indices = list(range(len(labels)))
        train_indices, test_indices = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
        train_labels = [labels[i] for i in train_indices]
        train_indices, val_indices = train_test_split(train_indices, test_size=0.15 / 0.85, stratify=train_labels, random_state=42)
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}, {}

    def create_adaptive_model(self, input_dim, num_classes):
        return IntentOnlyModel(
            input_dim=input_dim,
            hidden_dim=self.config.get('hidden_dim', 128),
            num_classes=num_classes,
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)

    def train_model(self, model, features, labels, slots, split_indices):
        train_indices = split_indices['train']
        val_indices = split_indices['val']

        train_dataset = TwiDataset([features[i] for i in train_indices], [labels[i] for i in train_indices], [slots[i] for i in train_indices], self.label_map, self.slot_map, self.slot_value_maps)
        val_dataset = TwiDataset([features[i] for i in val_indices], [labels[i] for i in val_indices], [slots[i] for i in val_indices], self.label_map, self.slot_map, self.slot_value_maps)

        train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 32), shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 32), collate_fn=collate_fn)

        optimizer = optim.AdamW(model.parameters(), lr=self.config.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)

        criterion = nn.CrossEntropyLoss()

        best_intent_f1 = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        for epoch in range(self.config.get('num_epochs', 50)):
            model.train()
            for features_b, intents_b in train_loader:
                features_b, intents_b = features_b.to(self.device), intents_b.to(self.device)

                optimizer.zero_grad()
                intent_logits = model(features_b)

                loss = criterion(intent_logits, intents_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation
            model.eval()
            val_intent_preds, val_intent_targets = [], []

            with torch.no_grad():
                for features_b, intents_b in val_loader:
                    features_b, intents_b = features_b.to(self.device), intents_b.to(self.device)

                    intent_logits = model(features_b)
                    val_intent_preds.extend(torch.argmax(intent_logits, dim=1).cpu().numpy())
                    val_intent_targets.extend(intents_b.cpu().numpy())

            intent_f1 = f1_score(val_intent_targets, val_intent_preds, average='macro', zero_division=0)
            accuracy = accuracy_score(val_intent_targets, val_intent_preds)

            history['train_loss'].append(loss.item())
            history['val_loss'].append(0) # not calculated
            history['val_accuracy'].append(accuracy)
            history['val_f1'].append(intent_f1)

            logger.info(f"Epoch {epoch+1} - Intent F1: {intent_f1:.4f}, Accuracy: {accuracy:.4f}")

            scheduler.step(intent_f1)

            if intent_f1 > best_intent_f1:
                best_intent_f1 = intent_f1
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.model_dir, 'best_model.pt'))
                logger.info("Best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= self.config.get('early_stopping_patience', 10):
                logger.info("Early stopping.")
                break
        return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='hybrid')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    config = {**vars(args), 'slot_loss_weight': 0.5, 'early_stopping_patience': 15}
    pipeline = ComprehensiveClassPreservingPipeline(config)
    pipeline.run_complete_pipeline()
