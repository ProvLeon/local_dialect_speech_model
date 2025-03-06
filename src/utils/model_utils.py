# src/utils/model_utils.py
import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)

def load_label_map(label_map_path):
    """
    Load label map from file (supports both .npy and .json formats)

    Args:
        label_map_path: Path to label map file

    Returns:
        Dictionary mapping labels to indices
    """
    if not os.path.exists(label_map_path):
        logger.error(f"Label map file not found: {label_map_path}")
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")

    # Handle different file formats
    if label_map_path.endswith('.npy'):
        try:
            return np.load(label_map_path, allow_pickle=True).item()
        except Exception as e:
            logger.error(f"Error loading .npy label map: {e}")
            raise
    elif label_map_path.endswith('.json'):
        try:
            with open(label_map_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading .json label map: {e}")
            raise
    else:
        raise ValueError(f"Unsupported label map format: {label_map_path}")

def get_model_input_dim(model_path):
    """
    Try to determine the input dimension for a model from its saved files

    Args:
        model_path: Path to model file

    Returns:
        Input dimension if found, else default value
    """
    # Try to get from model info file
    model_dir = os.path.dirname(model_path)
    model_info_path = os.path.join(model_dir, 'model_info.json')
    feature_info_path = os.path.join(model_dir, 'feature_info.json')

    # Check model_info.json first
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                if 'input_dim' in model_info:
                    return model_info['input_dim']
        except Exception as e:
            logger.warning(f"Could not read model_info.json: {e}")

    # Check feature_info.json next
    if os.path.exists(feature_info_path):
        try:
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                if 'input_dim' in feature_info:
                    return feature_info['input_dim']
        except Exception as e:
            logger.warning(f"Could not read feature_info.json: {e}")

    # Default value
    return 94  # Default input dimension for combined features
