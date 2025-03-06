import argparse
import logging
from train_enhanced_model import train_enhanced_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an enhanced Twi speech recognition model")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed data")
    parser.add_argument("--model-dir", type=str, default="data/models/enhanced", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs to train")

    args = parser.parse_args()

    logger.info("Starting enhanced model training...")
    model, trainer, test_acc = train_enhanced_model(args.data_dir, args.model_dir, args.epochs)
    logger.info(f"Training complete! Final test accuracy: {test_acc:.2f}%")
