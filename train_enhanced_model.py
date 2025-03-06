import os
import sys
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt  # Add this import at the top of the file
from src.models.speech_model import EnhancedTwiSpeechModel, AdvancedTrainer
from src.features.augmented_dataset import AugmentedTwiDataset
from src.utils.training_pipeline import TrainingPipeline
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_enhanced_model(data_dir="data/processed", model_dir="data/models/enhanced", epochs=100):
    """Train an enhanced model with all the improvements"""
    # 1. Configure the model with optimized parameters
    config = {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'batch_size': 64,
        'learning_rate': 0.002,
        'num_epochs': epochs,
        'early_stopping_patience': 15,
        'hidden_dim': 128,
        'dropout': 0.3,
        'num_heads': 8,
        'weight_decay': 0.01,
        'clip_grad_norm': 1.0,
        'accumulation_steps': 2,
        'random_seed': 42
    }

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])
        torch.backends.cudnn.deterministic = True

    # 2. Load data using existing pipeline
    pipeline = TrainingPipeline(config)
    features, labels, label_map = pipeline.load_data()

    # 3. Create dataset with augmentation
    augmented_dataset = AugmentedTwiDataset(
        features=features,
        labels=labels,
        label_to_idx=label_map,
        augment=True
    )

    # 4. Get class weights for balancing
    class_weights = augmented_dataset.get_class_weights()

    # 5. Create train/val/test split
    indices = np.arange(len(augmented_dataset))
    label_indices = [augmented_dataset.label_to_idx[label] for label in labels]

    # Split with stratification
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=label_indices, random_state=42
    )
    val_indices, test_indices = train_test_split(
        test_indices, test_size=0.5, stratify=[label_indices[i] for i in test_indices], random_state=42
    )

    # Create subset datasets
    train_dataset = Subset(augmented_dataset, train_indices)
    val_dataset = Subset(augmented_dataset, val_indices)
    test_dataset = Subset(augmented_dataset, test_indices)

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")

    # 6. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # Update steps per epoch for scheduler
    config['steps_per_epoch'] = len(train_loader)

    # 7. Initialize the enhanced model
    if len(features) > 0:
        input_dim = features[0].shape[0]
        logger.info(f"Using input dimension: {input_dim}")
    else:
        input_dim = 94  # Default
        logger.warning(f"No features found, using default input_dim={input_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = EnhancedTwiSpeechModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_classes=len(label_map),
        dropout=config['dropout'],
        num_heads=config['num_heads']
    )

    # 8. Initialize the trainer with advanced features
    trainer = AdvancedTrainer(model, device, config)

    # Set criterion with class weights for balancing
    if class_weights is not None:
        trainer.criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=0.1
        )

    # 9. Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        # Train one epoch
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)

        # Evaluate on validation set
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = trainer.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Update trainer history
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_acc'].append(val_acc)

        # Print progress
        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"LR: {trainer.optimizer.param_groups[0]['lr']:.6f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            # Save the best model
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config
                },
                os.path.join(model_dir, "best_model.pt")
            )

            logger.info(f"Validation improved by {improvement:.6f}. Best model saved!")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation. Patience: {patience_counter}/{config['early_stopping_patience']}")

            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or epoch == config['num_epochs'] - 1:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict() if hasattr(trainer, 'scheduler') else None,
                    'history': trainer.history,
                    'config': config
                },
                os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            )

    # 10. Load the best model for evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 11. Final evaluation on test set
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = trainer.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * test_correct / test_total

    logger.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 12. Generate confusion matrix and classification report
    try:
        from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
        # import matplotlib.pyplot as plt

        # Get class names
        idx_to_label = {v: k for k, v in label_map.items()}
        class_names = [idx_to_label.get(i, f"Unknown-{i}") for i in range(len(label_map))]

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Plot and save confusion matrix
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))

        # Generate and save classification report
        report = classification_report(all_targets, all_preds, target_names=class_names)
        logger.info(f"Classification Report:\n{report}")

        with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

    except ImportError:
        logger.warning("scikit-learn not installed. Skipping confusion matrix and classification report.")

    # 13. Save training history plot
    plt.figure(figsize=(15, 12))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(trainer.history['train_loss'], label='Train')
    plt.plot(trainer.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(trainer.history['train_acc'], label='Train')
    plt.plot(trainer.history['val_acc'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(trainer.history['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(alpha=0.3)

    # Add summary text
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Format summary text
    summary_text = (
        f"Training Summary:\n\n"
        f"Total Epochs: {len(trainer.history['train_loss'])}\n\n"
        f"Final Metrics:\n"
        f"  Train Loss: {trainer.history['train_loss'][-1]:.4f}\n"
        f"  Validation Loss: {trainer.history['val_loss'][-1]:.4f}\n"
        f"  Train Accuracy: {trainer.history['train_acc'][-1]:.2f}%\n"
        f"  Validation Accuracy: {trainer.history['val_acc'][-1]:.2f}%\n"
        f"  Test Accuracy: {test_acc:.2f}%\n\n"
        f"Best Validation:\n"
        f"  Loss: {best_val_loss:.4f}\n\n"
        f"Model Information:\n"
        f"  Input Dimension: {input_dim}\n"
        f"  Hidden Dimension: {config['hidden_dim']}\n"
        f"  Number of Classes: {len(label_map)}"
    )

    plt.text(0.05, 0.95, summary_text, va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()

    # 14. Save model information
    model_info = {
        'input_dim': input_dim,
        'hidden_dim': config['hidden_dim'],
        'num_classes': len(label_map),
        'model_type': 'EnhancedTwiSpeechModel',
        'final_metrics': {
            'train_loss': float(trainer.history['train_loss'][-1]),
            'val_loss': float(trainer.history['val_loss'][-1]),
            'train_acc': float(trainer.history['train_acc'][-1]),
            'val_acc': float(trainer.history['val_acc'][-1]),
            'test_acc': float(test_acc)
        },
        'best_val_loss': float(best_val_loss),
        'training_config': config,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

    with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Training complete! Model and statistics saved to {model_dir}")

    return model, trainer, test_acc

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train enhanced Twi speech recognition model")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed data")
    parser.add_argument("--model-dir", type=str, default="data/models/enhanced", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs to train")

    args = parser.parse_args()

    train_enhanced_model(args.data_dir, args.model_dir, args.epochs)
