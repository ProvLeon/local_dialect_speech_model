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
        """Load dataset from processed files with enhanced processing"""
        logger.info(f"Loading data from {self.data_dir}...")

        # Load features and labels
        features = np.load(os.path.join(self.data_dir, "features.npy"), allow_pickle=True)
        labels = np.load(os.path.join(self.data_dir, "labels.npy"), allow_pickle=True)

        # Load label map
        label_map_path = os.path.join(self.data_dir, "label_map.npy")
        if os.path.exists(label_map_path):
            label_map = np.load(label_map_path, allow_pickle=True).item()
        else:
            # Create label map if not exists
            unique_labels = sorted(set(labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            np.save(label_map_path, label_map)

        # Clean data
        features, labels = self._clean_data(features, labels)

        # Normalize features globally
        # Comment this out if you want to keep your existing normalization
        # features = self._normalize_features(features)

        logger.info(f"Loaded {len(features)} samples with {len(label_map)} classes")

        # Print feature dimensions
        if len(features) > 0:
            logger.info(f"Feature shape: {features[0].shape}")
            # Save feature metadata
            self._save_feature_metadata(features[0].shape[0], len(label_map))

        # Print class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info("Class distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  {label}: {count}")

        self.analyze_features(features)

        return features, labels, label_map


    def create_datasets(self, features, labels, label_map):
        """Create train and validation datasets"""
        # Create dataset
        dataset = TwiDataset(features, labels, label_map)

        # Get indices for splitting
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
        """Create data loaders for training"""
        batch_size = self.config.get('batch_size', 32)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
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

    def evaluate_model(self, trainer, val_loader):
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
                f.write(report)

        except ImportError:
            logger.warning("scikit-learn not available, skipping confusion matrix and classification report")

        # Calculate per-class accuracy
        class_correct = {}
        class_total = {}

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
        features, labels, label_map = self.load_data()

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
        train_dataset, val_dataset = self.create_datasets(features, labels, label_map)

        # 3. Create data loaders
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)

        # 4. Initialize model with correct input dimensions
        num_classes = len(label_map)
        model = self.initialize_model(input_dim, num_classes)

        # 5. Train model
        trainer, history = self.train_model(model, train_loader, val_loader)

        # 6. Evaluate model
        val_acc, class_correct, class_total = self.evaluate_model(trainer, val_loader)

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


# class EnhancedTrainingPipeline:
#     """Enhanced training pipeline with better data management and validation"""
#     def __init__(self, config):
#         self.config = config
#         self.data_dir = config.get('data_dir', 'data/processed')
#         self.model_dir = config.get('model_dir', 'data/models')

#         # Ensure directories exist
#         os.makedirs(self.model_dir, exist_ok=True)

#         # Set device
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         logger.info(f"Using device: {self.device}")

#         # Set random seed for reproducibility
#         seed = config.get('random_seed', 42)
#         self._set_seed(seed)

#     def _set_seed(self, seed):
#         """Set random seed for reproducibility"""
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)
#             # For deterministic behavior in CuDNN
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False

#     def load_and_preprocess_data(self):
#         """Enhanced data loading with feature normalization and validation"""
#         logger.info(f"Loading data from {self.data_dir}...")

#         # Load features and labels
#         features = np.load(os.path.join(self.data_dir, "features.npy"), allow_pickle=True)
#         labels = np.load(os.path.join(self.data_dir, "labels.npy"), allow_pickle=True)

#         # Load label map
#         label_map_path = os.path.join(self.data_dir, "label_map.npy")
#         if os.path.exists(label_map_path):
#             label_map = np.load(label_map_path, allow_pickle=True).item()
#         else:
#             unique_labels = sorted(set(labels))
#             label_map = {label: i for i, label in enumerate(unique_labels)}
#             np.save(label_map_path, label_map)

#         # Analyze and clean data
#         features, labels = self._clean_data(features, labels)

#         # Normalize features globally (across all samples)
#         features = self._normalize_features(features)

#         # Extract input dimension
#         input_dim = features[0].shape[0]
#         logger.info(f"Feature dimension: {input_dim}")

#         # Update config with exact feature dimension
#         self.config['input_dim'] = input_dim

#         # Save input dimension and other metadata
#         self._save_feature_metadata(input_dim, len(label_map))

#         return features, labels, label_map


#     def train_enhanced_model():
#         """Train an enhanced model with all improvements integrated"""
#         # Load configuration
#         from config.model_config import MODEL_CONFIG

#         # Enhance the config with better defaults
#         enhanced_config = MODEL_CONFIG.copy()
#         enhanced_config.update({
#             'batch_size': 64,  # Larger batch size
#             'learning_rate': 2e-3,  # Higher initial LR with OneCycle policy
#             'weight_decay': 0.01,  # L2 regularization
#             'num_epochs': 100,  # Train longer with early stopping
#             'early_stopping_patience': 15,  # More patience
#             'early_stopping_min_delta': 0.001,  # Minimum improvement
#             'clip_grad_norm': 1.0,  # Gradient clipping
#             'accumulation_steps': 2,  # Gradient accumulation
#             'dropout': 0.3,  # Reduced dropout for transformer
#             'random_seed': 42,  # For reproducibility
#             'model_type': 'enhanced'  # Use our enhanced model
#         })

#         # Initialize pipeline
#         pipeline = EnhancedTrainingPipeline(enhanced_config)

#         # Load and preprocess data
#         features, labels, label_map = pipeline.load_and_preprocess_data()

#         # Create datasets and loaders
#         train_loader, val_loader, test_loader = pipeline.create_datasets_and_loaders(
#             features, labels, label_map
#         )

#         # Initialize model with correct dimensions
#         input_dim = enhanced_config['input_dim']
#         num_classes = len(label_map)

#         model = EnhancedTwiSpeechModel(
#             input_dim=input_dim,
#             hidden_dim=enhanced_config.get('hidden_dim', 128),
#             num_classes=num_classes,
#             num_layers=enhanced_config.get('num_layers', 2),
#             dropout=enhanced_config.get('dropout', 0.3),
#             num_heads=enhanced_config.get('num_heads', 4)
#         )

#         # Initialize trainer
#         trainer = AdvancedTrainer(model, pipeline.device, enhanced_config)

#         # Train model with enhanced process
#         best_model = train_with_early_stopping(
#             trainer, model, train_loader, val_loader, enhanced_config
#         )

#         # Evaluate on test set
#         test_loss, test_acc = evaluate_model(best_model, test_loader, trainer.criterion, pipeline.device)
#         logger.info(f"Test accuracy: {test_acc:.2f}%")

#         # Save final model and results
#         save_results(best_model, trainer, test_acc, enhanced_config, label_map)

#         return best_model, trainer

#     def train_with_early_stopping(trainer, model, train_loader, val_loader, config):
#         """Train model with early stopping and proper validation"""
#         epochs = config.get('num_epochs', 100)
#         patience = config.get('early_stopping_patience', 15)
#         min_delta = config.get('early_stopping_min_delta', 0.001)
#         model_dir = config.get('model_dir', 'data/models')

#         best_val_loss = float('inf')
#         patience_counter = 0
#         best_model_state = None

#         for epoch in range(epochs):
#             # Train one epoch
#             train_loss, train_acc = trainer.train_epoch(train_loader, epoch)

#             # Validate
#             val_loss, val_acc = evaluate_model(
#                 model, val_loader, trainer.criterion, trainer.device
#             )

#             # Update history
#             trainer.history['val_loss'].append(val_loss)
#             trainer.history['val_acc'].append(val_acc)

#             # Print progress
#             logger.info(
#                 f"Epoch {epoch+1}/{epochs} - "
#                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
#                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
#                 f"LR: {trainer.optimizer.param_groups[0]['lr']:.6f}"
#             )

#             # Check for improvement
#             if val_loss < best_val_loss - min_delta:
#                 logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
#                 best_val_loss = val_loss
#                 patience_counter = 0
#                 best_model_state = model.state_dict().copy()

#                 # Save best model
#                 torch.save(
#                     {
#                         'epoch': epoch,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': trainer.optimizer.state_dict(),
#                         'val_loss': val_loss,
#                         'val_acc': val_acc
#                     },
#                     os.path.join(model_dir, "best_model.pt")
#                 )
#             else:
#                 patience_counter += 1
#                 logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

#                 # Early stopping
#                 if patience_counter >= patience:
#                     logger.info(f"Early stopping triggered after {epoch+1} epochs")
#                     break

#             # Save checkpoint periodically
#             if (epoch + 1) % 10 == 0:
#                 torch.save(
#                     {
#                         'epoch': epoch,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': trainer.optimizer.state_dict(),
#                         'scheduler_state_dict': trainer.scheduler.state_dict(),
#                         'history': trainer.history
#                     },
#                     os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
#                 )

#         # Load best model state
#         if best_model_state is not None:
#             model.load_state_dict(best_model_state)

#         # Plot training history
#         plot_training_history(trainer.history, os.path.join(model_dir, 'training_history.png'))

#         return model

#     def evaluate_model(model, data_loader, criterion, device):
#         """Evaluate model on data loader"""
#         model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for inputs, targets in data_loader:
#                 inputs, targets = inputs.to(device), targets.to(device)

#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)

#                 total_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()

#         avg_loss = total_loss / len(data_loader)
#         accuracy = 100.0 * correct / total

#         return avg_loss, accuracy

#     def plot_training_history(history, save_path):
#         """Create enhanced training history plot"""
#         plt.figure(figsize=(15, 12))

#         # 1. Training & Validation Loss
#         plt.subplot(2, 2, 1)
#         plt.plot(history['train_loss'], label='Training')
#         plt.plot(history['val_loss'], label='Validation')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Loss Curves')
#         plt.legend()
#         plt.grid(alpha=0.3)

#         # 2. Training & Validation Accuracy
#         plt.subplot(2, 2, 2)
#         plt.plot(history['train_acc'], label='Training')
#         plt.plot(history['val_acc'], label='Validation')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy (%)')
#         plt.title('Accuracy Curves')
#         plt.legend()
#         plt.grid(alpha=0.3)

#         # 3. Learning Rate
#         plt.subplot(2, 2, 3)
#         plt.plot(history['learning_rates'])
#         plt.xlabel('Epoch')
#         plt.ylabel('Learning Rate')
#         plt.title('Learning Rate Schedule')
#         plt.grid(alpha=0.3)

#         # 4. Training Progress Summary
#         plt.subplot(2, 2, 4)
#         plt.axis('off')

#         # Calculate metrics for summary
#         final_train_loss = history['train_loss'][-1]
#         final_val_loss = history['val_loss'][-1]
#         best_val_loss = min(history['val_loss'])
#         best_epoch = history['val_loss'].index(best_val_loss) + 1

#         final_train_acc = history['train_acc'][-1]
#         final_val_acc = history['val_acc'][-1]
#         best_val_acc = max(history['val_acc'])

#         summary_text = (
#             f"Training Summary:\n\n"
#             f"Total Epochs: {len(history['train_loss'])}\n\n"
#             f"Final Metrics:\n"
#             f"  Training Loss: {final_train_loss:.4f}\n"
#             f"  Validation Loss: {final_val_loss:.4f}\n"
#             f"  Training Accuracy: {final_train_acc:.2f}%\n"
#             f"  Validation Accuracy: {final_val_acc:.2f}%\n\n"
#             f"Best Validation Performance:\n"
#             f"  Best Loss: {best_val_loss:.4f} (Epoch {best_epoch})\n"
#             f"  Best Accuracy: {best_val_acc:.2f}%\n\n"
#             f"Learning Rate:\n"
#             f"  Initial: {history['learning_rates'][0]:.6f}\n"
#             f"  Final: {history['learning_rates'][-1]:.6f}"
#         )

#         plt.text(0.1, 0.5, summary_text, fontsize=11)

#         plt.tight_layout()
#         plt.savefig(save_path, dpi=200)
#         plt.close()

#     def save_results(self, model, trainer, test_acc, config, label_map):
#         """Save model, configuration, and results"""
#         model_dir = config.get('model_dir', 'data/models')

#         # 1. Save final model
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'config': config,
#             'history': trainer.history,
#             'test_accuracy': test_acc
#         }, os.path.join(model_dir, "final_model.pt"))

#         # 2. Save model architecture details
#         model_info = {
#             'input_dim': config['input_dim'],
#             'hidden_dim': config.get('hidden_dim', 128),
#             'num_classes': len(label_map),
#             'model_type': config.get('model_type', 'enhanced'),
#             'test_accuracy': float(test_acc),
#             'parameters': sum(p.numel() for p in model.parameters()),
#             'date_trained': time.strftime("%Y-%m-%d %H:%M:%S")
#         }

#         with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
#             json.dump(model_info, f, indent=2)

#     def _normalize_features(self, features):
#         """Normalize features across all samples for better convergence"""
#         # Stack all features to compute global statistics
#         all_features = np.vstack([feat.reshape(feat.shape[0], -1) for feat in features])

#         # Compute global mean and std
#         global_mean = np.mean(all_features, axis=1, keepdims=True)
#         global_std = np.std(all_features, axis=1, keepdims=True) + 1e-5

#         # Normalize each sample
#         normalized_features = []
#         for feat in features:
#             feat_reshaped = feat.reshape(feat.shape[0], -1)
#             normalized = (feat_reshaped - global_mean) / global_std
#             normalized_features.append(normalized.reshape(feat.shape))

#         return normalized_features

#     def _clean_data(self, features, labels):
#         """Clean data by removing invalid samples"""
#         valid_indices = []

#         for i, (feat, label) in enumerate(zip(features, labels)):
#             # Check for NaN or Inf
#             if np.isnan(feat).any() or np.isinf(feat).any():
#                 logger.warning(f"Sample {i} contains NaN or Inf values. Skipping.")
#                 continue

#             # Check for empty or very small features
#             if feat.size == 0 or np.prod(feat.shape) < 10:
#                 logger.warning(f"Sample {i} has invalid shape {feat.shape}. Skipping.")
#                 continue

#             valid_indices.append(i)

#         logger.info(f"Keeping {len(valid_indices)}/{len(features)} valid samples")

#         return [features[i] for i in valid_indices], [labels[i] for i in valid_indices]

#     def _save_feature_metadata(self, input_dim, num_classes):
#         """Save feature metadata for model initialization"""
#         metadata = {
#             'input_dim': int(input_dim),
#             'num_classes': int(num_classes),
#             'preprocessing_info': {
#                 'normalization': 'global_mean_std',
#                 'cleaning': 'remove_nan_inf'
#             }
#         }

#         with open(os.path.join(self.model_dir, 'feature_metadata.json'), 'w') as f:
#             json.dump(metadata, f, indent=2)

#     def create_datasets_and_loaders(self, features, labels, label_map):
#         """Create datasets and data loaders with better balancing"""
#         # Create dataset
#         dataset = TwiDataset(features, labels, label_map)

#         # Get indices for splitting with stratification
#         indices = np.arange(len(dataset))
#         label_indices = np.array([dataset.label_to_idx[label] for label in labels])

#         # Create train, validation, and test splits
#         train_indices, test_indices = train_test_split(
#             indices, test_size=0.2, stratify=label_indices, random_state=42
#         )

#         # Further split test into validation and test
#         val_indices, test_indices = train_test_split(
#             test_indices, test_size=0.5, stratify=label_indices[test_indices], random_state=42
#         )

#         # Create subset datasets
#         train_dataset = Subset(dataset, train_indices)
#         val_dataset = Subset(dataset, val_indices)
#         test_dataset = Subset(dataset, test_indices)

#         logger.info(f"Train set: {len(train_dataset)} samples")
#         logger.info(f"Validation set: {len(val_dataset)} samples")
#         logger.info(f"Test set: {len(test_dataset)} samples")

#         # Create data loaders
#         batch_size = self.config.get('batch_size', 32)

#         # Apply transforms to augment training data
#         train_transform = self._get_train_transforms()

#         # Create data loaders with appropriate settings
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True if self.device.type == 'cuda' else False,
#         )

#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True if self.device.type == 'cuda' else False
#         )

#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True if self.device.type == 'cuda' else False
#         )

#         self.config['steps_per_epoch'] = len(train_loader)

#         return train_loader, val_loader, test_loader

#     def _get_train_transforms(self):
#         """Create transforms for data augmentation during training"""
#         # These transforms would be applied to your feature tensors
#         # For audio features, typical augmentations include:
#         # - SpecAugment (time/frequency masking)
#         # - Random time shifting
#         # - Feature dropout
#         # Implementation depends on your specific preprocessing
#         return None  # Replace with actual transforms
