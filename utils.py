"""
Utility functions module.

This module contains various utility functions used throughout the application.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
from datetime import datetime
import hashlib
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Utils:
    """Utility class with common helper functions."""
    
    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            directory: Path object for the directory
        """
        directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def load_json_file(file_path: Path) -> Dict[str, Any]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with data
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return {}
    
    @staticmethod
    def save_json_file(data: Dict[str, Any], file_path: Path) -> bool:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            file_path: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {e}")
            return False
    
    @staticmethod
    def load_yaml_file(file_path: Path) -> Dict[str, Any]:
        """
        Load data from a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dictionary with data
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            return {}
    
    @staticmethod
    def save_yaml_file(data: Dict[str, Any], file_path: Path) -> bool:
        """
        Save data to a YAML file.
        
        Args:
            data: Data to save
            file_path: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Error saving YAML file {file_path}: {e}")
            return False
    
    @staticmethod
    def hash_string(text: str) -> str:
        """
        Generate a hash for a string.
        
        Args:
            text: Input string
            
        Returns:
            SHA-256 hash
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    @staticmethod
    def get_file_hash(file_path: Path) -> Optional[str]:
        """
        Generate hash for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash or None if error
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp as string.
        
        Returns:
            Timestamp string (YYYY-MM-DD HH:MM:SS)
        """
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def get_date() -> str:
        """
        Get current date as string.
        
        Returns:
            Date string (YYYY-MM-DD)
        """
        return datetime.now().strftime('%Y-%m-%d')
    
    @staticmethod
    def get_time() -> str:
        """
        Get current time as string.
        
        Returns:
            Time string (HH:MM:SS)
        """
        return datetime.now().strftime('%H:%M:%S')
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if an image is valid.
        
        Args:
            image: Image array
            
        Returns:
            True if valid, False otherwise
        """
        if image is None:
            return False
        if image.size == 0:
            return False
        if len(image.shape) not in [2, 3]:
            return False
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False
        return True
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: tuple) -> Optional[np.ndarray]:
        """
        Resize an image.
        
        Args:
            image: Input image
            target_size: (width, height) tuple
            
        Returns:
            Resized image or None if error
        """
        try:
            if not Utils.validate_image(image):
                return None
            return cv2.resize(image, target_size)
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return None
    
    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert image to RGB.
        
        Args:
            image: Input image
            
        Returns:
            RGB image or None if error
        """
        try:
            if not Utils.validate_image(image):
                return None
            
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return image
        except Exception as e:
            logger.error(f"Error converting to RGB: {e}")
            return None
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """
        Format confidence score as percentage.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Formatted string (e.g., "95.5%")
        """
        return f"{confidence*100:.1f}%"
    
    @staticmethod
    def create_progress_bar(progress: float, width: int = 50) -> str:
        """
        Create a text progress bar.
        
        Args:
            progress: Progress value (0-1)
            width: Width of the progress bar
            
        Returns:
            Progress bar string
        """
        filled = int(progress * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {progress*100:.1f}%"
    
    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers.
        
        Args:
            a: Numerator
            b: Denominator
            default: Default value if division fails
            
        Returns:
            Division result or default
        """
        try:
            if b == 0:
                return default
            return a / b
        except:
            return default


# Create a singleton instance
utils = Utils()


if __name__ == "__main__":
    # Test utilities
    print("Testing Utilities")
    print("=" * 50)
    
    print(f"Timestamp: {utils.get_timestamp()}")
    print(f"Date: {utils.get_date()}")
    print(f"Time: {utils.get_time()}")
    print(f"Progress: {utils.create_progress_bar(0.75)}")
    print(f"Confidence: {utils.format_confidence(0.956)}")