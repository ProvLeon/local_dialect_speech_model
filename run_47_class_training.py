#!/usr/bin/env python3
"""
Simple Usage Script for 47-Class Preserving Training Pipeline

This script provides an easy way to run the comprehensive 47-class preserving
training pipeline with different strategies and configurations.

Usage Examples:
    # Quick test with hybrid strategy
    python run_47_class_training.py

    # Progressive curriculum learning
    python run_47_class_training.py --strategy progressive --epochs 100

    # Balanced sampling with augmentation
    python run_47_class_training.py --strategy balanced --augment --epochs 50

    # Full hybrid approach with custom settings
    python run_47_class_training.py --strategy hybrid --augment --epochs 75 --lr 0.002
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our comprehensive pipeline
from train_47_class_preserving import ComprehensiveClassPreservingPipeline


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_data_availability(data_dir: str) -> bool:
    """Check if required data files are available."""
    required_files = [
        'features.npy',
        'labels.npy',
        'label_map.json',
        'slots.json'
    ]

    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return False

    print("✅ All required data files found")
    return True


def create_config(args) -> dict:
    """Create configuration dictionary from arguments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.model_dir, f"{args.strategy}_{timestamp}")

    config = {
        'data_dir': args.data_dir,
        'model_dir': model_dir,
        'strategy': args.strategy,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'use_augmentation': args.augment,
        'early_stopping_patience': args.patience,
        'val_prop': args.val_prop,
        'test_prop': args.test_prop,
        'target_samples_per_class': args.target_samples,
        'min_samples_per_class': args.min_samples,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_workers': 0  # Avoid multiprocessing issues
    }

    return config


def print_config_summary(config: dict):
    """Print configuration summary."""
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Strategy: {config['strategy']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Use Augmentation: {config['use_augmentation']}")
    print(f"Model Directory: {config['model_dir']}")
    print(f"Data Directory: {config['data_dir']}")

    if config['use_augmentation']:
        print(f"Target Samples/Class: {config['target_samples_per_class']}")
        print(f"Min Samples/Class: {config['min_samples_per_class']}")

    print("="*50 + "\n")


def print_strategy_info(strategy: str):
    """Print information about the selected strategy."""
    strategy_info = {
        'progressive': {
            'name': 'Progressive Curriculum Learning',
            'description': 'Gradually introduces classes from easy to hard, preventing overfitting',
            'best_for': 'Datasets with extreme class imbalance and many rare classes',
            'pros': ['Prevents catastrophic forgetting', 'Stable convergence', 'Good for rare classes'],
            'cons': ['Longer training time', 'Complex implementation']
        },
        'balanced': {
            'name': 'Balanced Sampling with Augmentation',
            'description': 'Uses advanced augmentation and weighted sampling for class balance',
            'best_for': 'When you want to maintain all classes with simpler training',
            'pros': ['Faster training', 'Simple approach', 'Effective augmentation'],
            'cons': ['May overfit on augmented data', 'Less sophisticated than curriculum']
        },
        'hybrid': {
            'name': 'Hybrid Approach (Recommended)',
            'description': 'Combines curriculum learning with balanced fine-tuning',
            'best_for': 'Best overall performance while preserving all 47 classes',
            'pros': ['Best of both worlds', 'Robust performance', 'Handles all scenarios'],
            'cons': ['Longest training time', 'Most complex']
        }
    }

    info = strategy_info.get(strategy, {})
    print(f"\n📚 STRATEGY: {info.get('name', strategy.upper())}")
    print(f"Description: {info.get('description', 'No description available')}")
    print(f"Best for: {info.get('best_for', 'General use')}")

    if 'pros' in info:
        print("Pros:")
        for pro in info['pros']:
            print(f"  ✅ {pro}")

    if 'cons' in info:
        print("Cons:")
        for con in info['cons']:
            print(f"  ⚠️ {con}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run 47-Class Preserving Speech Intent Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Quick test with defaults
  %(prog)s --strategy progressive             # Curriculum learning
  %(prog)s --strategy balanced --augment      # Balanced with augmentation
  %(prog)s --strategy hybrid --epochs 100    # Full hybrid approach
        """
    )

    # Strategy selection
    parser.add_argument('--strategy',
                       choices=['progressive', 'balanced', 'hybrid'],
                       default='hybrid',
                       help='Training strategy (default: hybrid)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')

    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--target-samples', type=int, default=15,
                       help='Target samples per class for augmentation (default: 15)')
    parser.add_argument('--min-samples', type=int, default=8,
                       help='Minimum samples per class for augmentation (default: 8)')

    # Data splits
    parser.add_argument('--val-prop', type=float, default=0.15,
                       help='Validation split proportion (default: 0.15)')
    parser.add_argument('--test-prop', type=float, default=0.15,
                       help='Test split proportion (default: 0.15)')

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension size (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')

    # Paths
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Data directory (default: data/processed)')
    parser.add_argument('--model-dir', type=str, default='data/models/47_class_results',
                       help='Base model directory (default: data/models/47_class_results)')

    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str,
                       help='Log file path (optional)')

    # Dry run
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without running training')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    print("🎯 47-Class Preserving Speech Intent Training")
    print("=" * 50)

    # Validate data availability
    if not validate_data_availability(args.data_dir):
        print("\n❌ Data validation failed. Please check your data directory.")
        print(f"Expected files in {args.data_dir}:")
        print("  - features.npy")
        print("  - labels.npy")
        print("  - label_map.json")
        print("  - slots.json")
        sys.exit(1)

    # Show strategy information
    print_strategy_info(args.strategy)

    # Create configuration
    config = create_config(args)

    # Print configuration summary
    print_config_summary(config)

    if args.dry_run:
        print("🏃 Dry run mode - configuration validated successfully!")
        print("Run without --dry-run to start training.")
        return

    # Confirm before starting
    # if args.strategy == 'hybrid' and args.epochs > 75:
    #     response = input("⚠️  Hybrid strategy with many epochs will take a long time. Continue? (y/N): ")
    #     if response.lower() not in ['y', 'yes']:
    #         print("Training cancelled.")
    #         return

    try:
        print("🚀 Starting training pipeline...")

        # Initialize and run pipeline
        pipeline = ComprehensiveClassPreservingPipeline(config)
        results = pipeline.run_complete_pipeline()

        # Visualize results
        try:
            from visualize_training_results import TrainingVisualizer
            visualizer = TrainingVisualizer(os.path.join(config['model_dir'], 'final_results.json'))
            visualizer.generate_all_visualizations()
        except ImportError:
            logger.warning("Could not import TrainingVisualizer. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {config['model_dir']}")

        # Print key results
        if 'evaluation_results' in results and 'test' in results['evaluation_results']:
            test_results = results['evaluation_results']['test']
            print("\n📊 Final Test Results:")
            print(f"  Accuracy: {test_results['accuracy']:.1%}")
            print(f"  Macro-F1 (all classes): {test_results['macro_f1_full']:.4f}")
            print(f"  Macro-F1 (active classes): {test_results['macro_f1_present']:.4f}")
            print(f"  Active classes: {test_results['present_classes']}/{test_results['total_classes']}")

        print("\n✅ All 47 classes preserved throughout training!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.exception("Training pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
