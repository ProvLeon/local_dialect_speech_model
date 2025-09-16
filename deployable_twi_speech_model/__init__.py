"""
Packaged Speech Model

Easy-to-use interface for speech intent recognition.
"""

import sys
from pathlib import Path

__version__ = "1.0.0"

# Convenience class
class SpeechModelPackage:
    @staticmethod
    def from_pretrained(package_path: str):
        """Load a pre-trained model package."""
        # Add utils directory to path
        package_path = Path(package_path)
        utils_path = package_path / 'utils'
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))

        from inference import ModelInference
        return ModelInference(package_path)
