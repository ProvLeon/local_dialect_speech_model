import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..features.feature_extractor import TwiDataset
from ..models.speech_model import ImprovedTwiSpeechModel, ImprovedTrainer
from ..preprocessing.enhanced_audio_processor import EnhancedAudioProcessor
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config):
        """
        Complete training pipeline from data to model evaluation

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = config.get('data_dir', 'data/processed')
        self.model_dir = config.get('model_dir', 'data/models')
        self.feature_type = config.get('feature_type', 'combined')

        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def analyze_features(self, features):
        """Analyze feature shapes and content for debugging"""
        if len(features) == 0:
            logger.warning("No features found to analyze")
            return

        logger.info(f"Feature analysis:")
        logger.info(f"  Number of samples: {len(features)}")
        logger.info(f"  Feature shape: {features[0].shape}")

        # Check for NaN or infinity values
        sample = features[0]
        logger.info(f"  Contains NaN: {np.isnan(sample).any()}")
        logger.info(f"  Contains Inf: {np.isinf(sample).any()}")

        # Check statistics
        logger.info(f"  Min value: {np.min(sample)}")
        logger.info(f"  Max value: {np.max(sample)}")
        logger.info(f"  Mean value: {np.mean(sample)}")
        logger.info(f"  Std deviation: {np.std(sample)}")

        # Print the shape of a few more samples to ensure consistency
        if len(features) > 1:
            logger.info(f"  Second sample shape: {features[1].shape}")
        if len(features) > 2:
            logger.info(f"  Third sample shape: {features[2].shape}")

        # Check if feature dimensions are consistent across all samples
        shapes = [f.shape for f in features]
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            logger.warning(f"Inconsistent feature shapes detected! Found {len(unique_shapes)} different shapes.")
            for i, shape in enumerate(unique_shapes):
                count = shapes.count(shape)
                logger.warning(f"  Shape {shape}: {count} samples")


    def load_data(self):
        """Load dataset from processed files with enhanced processing (now slot-aware)"""
        logger.info(f"Loading data from {self.data_dir}...")

        # Load features and labels
        features = np.load(os.path.join(self.data_dir, "features.npy"), allow_pickle=True)
        labels = np.load(os.path.join(self.data_dir, "labels.npy"), allow_pickle=True)

        # Attempt to load slots (optional)
        slots_path_json = os.path.join(self.data_dir, "slots.json")
        if os.path.exists(slots_path_json):
            try:
                with open(slots_path_json, "r") as f:
                    slots = json.load(f)
                if len(slots) != len(labels):
                    logger.warning(f"slots.json length {len(slots)} != labels length {len(labels)}. Padding/truncating.")
                    if len(slots) < len(labels):
                        slots.extend([{} for _ in range(len(labels) - len(slots))])
                    else:
                        slots = slots[:len(labels)]
            except Exception as e:
                logger.warning(f"Failed loading slots.json: {e}. Proceeding with empty slot dicts.")
                slots = [{} for _ in range(len(labels))]
        else:
            slots = [{} for _ in range(len(labels))]

        # Load label map (support legacy .npy or new .json)
        label_map_path_npy = os.path.join(self.data_dir, "label_map.npy")
        label_map_path_json = os.path.join(self.data_dir, "label_map.json")
        label_map = None
        if os.path.exists(label_map_path_json):
            try:
                with open(label_map_path_json, "r") as f:
                    label_map = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read label_map.json: {e}")
        if label_map is None and os.path.exists(label_map_path_npy):
            try:
                label_map = np.load(label_map_path_npy, allow_pickle=True).item()
            except Exception as e:
                logger.warning(f"Could not read label_map.npy: {e}")
        if label_map is None:
            unique_labels_sorted = sorted(set(labels))
            label_map = {label: i for i, label in enumerate(unique_labels_sorted)}
            with open(label_map_path_json, 'w') as f:
                json.dump(label_map, f, indent=2)

        # Clean data (features / labels) – slots must align with kept indices
        cleaned_features, cleaned_labels = [], []
        cleaned_slots = []
        for feat, lab, sl in zip(features, labels, slots):
            if np.isnan(feat).any() or np.isinf(feat).any() or feat.size == 0 or np.prod(feat.shape) < 10:
                continue
            cleaned_features.append(feat)
            cleaned_labels.append(lab)
            cleaned_slots.append(sl)

        features, labels, slots = cleaned_features, cleaned_labels, cleaned_slots

        logger.info(f"Loaded {len(features)} samples with {len(label_map)} classes")

        if len(features) > 0:
            logger.info(f"Feature shape: {features[0].shape}")
            self._save_feature_metadata(features[0].shape[0], len(label_map))

        unique_labels_arr, counts = np.unique(labels, return_counts=True)
        logger.info("Class distribution:")
        for label, count in zip(unique_labels_arr, counts):
            logger.info(f"  {label}: {count}")

        self.analyze_features(features)

        return features, labels, label_map, slots


    def create_datasets(self, features, labels, label_map, slots):
        """Create train and validation datasets (slot-aware)"""
        dataset = TwiDataset(features, labels, label_map, slots=slots)

        indices = np.arange(len(dataset))
        label_indices = np.array([dataset.label_to_idx[label] for label in labels])

        # Stratified split
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, stratify=label_indices, random_state=42
        )

        # Create subset datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        logger.info(f"Train set: {len(train_dataset)} samples")
        logger.info(f"Validation set: {len(val_dataset)} samples")

        return train_dataset, val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        """Create data loaders for training (strips slot dicts for legacy trainers)"""
        batch_size = self.config.get('batch_size', 32)

        def _collate_strip_slots(batch):
            import torch
            feats, labs = [], []
            for sample in batch:
                if len(sample) == 3:
                    f, l, _ = sample
                else:
                    f, l = sample
                feats.append(f)
                labs.append(l)
            return torch.stack(feats, 0), torch.stack(labs, 0)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=_collate_strip_slots
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=_collate_strip_slots
        )

        return train_loader, val_loader

    def initialize_model(self, input_dim, num_classes):
        """Initialize model architecture with the correct input dimension"""
        hidden_dim = self.config.get('hidden_dim', 128)
        num_layers = self.config.get('num_layers', 2)
        dropout = self.config.get('dropout', 0.5)

        # Print feature shape for debugging
        logger.info(f"Creating model with input_dim={input_dim}, num_classes={num_classes}")

        model = ImprovedTwiSpeechModel(
            input_dim=input_dim,  # This is crucial - use actual feature dimension
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout
        )


        return model


    def _normalize_features(self, features):
        """Normalize features across all samples for better convergence"""
        # Stack all features to compute global statistics
        all_features = np.vstack([feat.reshape(feat.shape[0], -1) for feat in features])

        # Compute global mean and std
        global_mean = np.mean(all_features, axis=1, keepdims=True)
        global_std = np.std(all_features, axis=1, keepdims=True) + 1e-5

        # Normalize each sample
        normalized_features = []
        for feat in features:
            feat_reshaped = feat.reshape(feat.shape[0], -1)
            normalized = (feat_reshaped - global_mean) / global_std
            normalized_features.append(normalized.reshape(feat.shape))

        return normalized_features

    def _clean_data(self, features, labels):
        """Clean data by removing invalid samples"""
        valid_indices = []

        for i, (feat, label) in enumerate(zip(features, labels)):
            # Check for NaN or Inf
            if np.isnan(feat).any() or np.isinf(feat).any():
                logger.warning(f"Sample {i} contains NaN or Inf values. Skipping.")
                continue

            # Check for empty or very small features
            if feat.size == 0 or np.prod(feat.shape) < 10:
                logger.warning(f"Sample {i} has invalid shape {feat.shape}. Skipping.")
                continue

            valid_indices.append(i)

        logger.info(f"Keeping {len(valid_indices)}/{len(features)} valid samples")

        return [features[i] for i in valid_indices], [labels[i] for i in valid_indices]

    def _save_feature_metadata(self, input_dim, num_classes):
        """Save feature metadata for model initialization"""
        metadata = {
            'input_dim': int(input_dim),
            'num_classes': int(num_classes),
            'preprocessing_info': {
                'normalization': 'global_mean_std',
                'cleaning': 'remove_nan_inf'
            }
        }

        with open(os.path.join(self.model_dir, 'feature_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


    def train_model(self, model, train_loader, val_loader):
        """Train the model"""
        learning_rate = self.config.get('learning_rate', 0.001)
        num_epochs = self.config.get('num_epochs', 50)
        patience = self.config.get('early_stopping_patience', 10)

        logger.info(f"Training model for up to {num_epochs} epochs with patience {patience}")

        trainer = ImprovedTrainer(
            model,
            self.device,
            learning_rate=learning_rate,
            model_dir=self.model_dir
        )

        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            early_stopping_patience=patience,
            class_weighting=True
        )

        return trainer, history

    def evaluate_model(self, trainer, val_loader, label_map):
        """Evaluate the model on validation data and generate metrics"""
        # Load best model
        best_model_path = os.path.join(self.model_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            trainer.load_model(best_model_path)
            logger.info("Loaded best model for evaluation")

        # Evaluate
        val_loss, val_acc, all_preds, all_targets = trainer.validate(val_loader)
        logger.info(f"Final validation accuracy: {val_acc:.2f}%")

        # Generate confusion matrix if scikit-learn is available
        try:
            from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

            # Get class names
            idx_to_label = {v: k for k, v in val_loader.dataset.dataset.label_to_idx.items()}
            class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

            # Compute confusion matrix
            cm = confusion_matrix(all_targets, all_preds)

            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'))

            # Generate classification report
            report = classification_report(all_targets, all_preds, target_names=class_names)
            logger.info(f"Classification Report:\n{report}")

            # Save report to file
            with open(os.path.join(self.model_dir, 'classification_report.txt'), 'w') as f:
                f.write(str(report))

        except ImportError:
            logger.warning("scikit-learn not available, skipping confusion matrix and classification report")

        # Calculate per-class accuracy
        class_correct = {}
        class_total = {}

        idx_to_label = {i: label for label, i in label_map.items()} if 'label_map' in locals() else {}
        for pred, true in zip(all_preds, all_targets):
            label = idx_to_label[true]
            if label not in class_total:
                class_total[label] = 0
                class_correct[label] = 0

            class_total[label] += 1
            if pred == true:
                class_correct[label] += 1

        # Print per-class accuracy
        logger.info("Per-class accuracy:")
        for label in sorted(class_total.keys()):
            acc = 100.0 * class_correct[label] / class_total[label]
            logger.info(f"  {label}: {acc:.2f}% ({class_correct[label]}/{class_total[label]})")

        return val_acc, class_correct, class_total

    def run(self):
        # 1. Load data
        features, labels, label_map, slots = self.load_data()

        # DEBUG: Print feature shape to understand dimensions
        if len(features) > 0:
            # Extract the feature dimension - this should be the first dimension (channels)
            input_dim = features[0].shape[0]
            logger.info(f"Feature shape detected: {features[0].shape}, using input_dim={input_dim}")

            # Make sure this is actually being used
            print(f"Setting input_dim to {input_dim} based on feature shape")
            self.config['input_dim'] = input_dim
        else:
            # Fallback
            input_dim = self.config.get('input_dim', 39)
            logger.warning(f"No features found, using default input_dim={input_dim}")


        # Save the input dimension for future reference
        import json
        model_info_path = os.path.join(self.model_dir, 'feature_info.json')
        with open(model_info_path, 'w') as f:
            json.dump({'input_dim': int(input_dim)}, f)

        # 2. Create datasets
        train_dataset, val_dataset = self.create_datasets(features, labels, label_map, slots)

        # 3. Create data loaders
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)

        # 4. Initialize model with correct input dimensions
        num_classes = len(label_map)
        model = self.initialize_model(input_dim, num_classes)

        # 5. Train model
        trainer, history = self.train_model(model, train_loader, val_loader)

        # 6. Evaluate model
        val_acc, class_correct, class_total = self.evaluate_model(trainer, val_loader, label_map)

        # 7. Save model info
        model_info = {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'class_mapping': label_map,
            'validation_accuracy': val_acc,
            'per_class_accuracy': {k: 100.0 * class_correct[k] / class_total[k] for k in class_total},
            'config': self.config
        }

        # Save as a metadata file for later reference
        import json
        with open(os.path.join(self.model_dir, 'model_info.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            model_info_serializable = {}
            for k, v in model_info.items():
                if k == 'per_class_accuracy':
                    model_info_serializable[k] = {str(c): float(a) for c, a in v.items()}
                elif k == 'class_mapping':
                    model_info_serializable[k] = {str(c): int(i) for c, i in v.items()}
                elif k == 'config':
                    model_info_serializable[k] = {k2: (v2 if not isinstance(v2, (np.integer, np.floating)) else float(v2))
                                                for k2, v2 in v.items()}
                else:
                    model_info_serializable[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v

            json.dump(model_info_serializable, f, indent=2)

        logger.info(f"Training pipeline complete. Model and metadata saved to {self.model_dir}")

        return model, trainer, history
