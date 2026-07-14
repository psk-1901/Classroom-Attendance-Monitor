"""
Configuration module for the Attendance System.

This module contains all configuration settings, paths, and constants
used throughout the application.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Main configuration class for the attendance system."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / 'data'
    DATASET_DIR = BASE_DIR / 'dataset'
    EMBEDDINGS_DIR = BASE_DIR / 'embeddings'
    ATTENDANCE_DIR = BASE_DIR / 'Attendance'
    UNKNOWN_FACES_DIR = BASE_DIR / 'unknown_faces'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, DATASET_DIR, EMBEDDINGS_DIR, ATTENDANCE_DIR,
                     UNKNOWN_FACES_DIR, MODELS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Face detection settings
    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold
    IMAGE_SIZE = (160, 160)  # FaceNet input size
    
    # Dataset collection settings
    IMAGES_PER_STUDENT = 100
    CAPTURE_INTERVAL = 0.1  # Seconds between captures
    BLUR_THRESHOLD = 100.0  # Lower = more blur detection
    
    # MongoDB settings
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'attendance_system')
    
    # Collections
    COLLECTIONS = {
        'students': 'students',
        'embeddings': 'embeddings',
        'attendance': 'attendance',
        'logs': 'logs'
    }
    
    # Attendance settings
    ATTENDANCE_SESSION_TIMEOUT = 300  # 5 minutes in seconds
    
    # Excel report settings
    EXCEL_TEMPLATE = {
        'college_name': 'Your College Name',
        'department': 'Department',
        'faculty': 'Faculty Name',
        'subject': 'Subject Name'
    }
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / 'attendance_system.log'
    
    # FaceNet model settings
    FACENET_MODEL_PATH = MODELS_DIR / 'facenet_model.h5'
    FACENET_WEIGHTS_PATH = MODELS_DIR / 'facenet_weights.h5'
    
    # Embedding storage
    EMBEDDINGS_FILE = EMBEDDINGS_DIR / 'embeddings.pkl'
    
    @classmethod
    def get_collection_name(cls, collection_key: str) -> str:
        """Get the collection name for a given key."""
        return cls.COLLECTIONS.get(collection_key, collection_key)
    
    @classmethod
    def get_attendance_config(cls) -> Dict[str, Any]:
        """Get attendance configuration settings."""
        return {
            'session_timeout': cls.ATTENDANCE_SESSION_TIMEOUT,
            'threshold': cls.FACE_RECOGNITION_THRESHOLD,
            'images_per_student': cls.IMAGES_PER_STUDENT
        }
    
    @classmethod
    def get_paths(cls) -> Dict[str, Path]:
        """Get all path configurations."""
        return {
            'base': cls.BASE_DIR,
            'dataset': cls.DATASET_DIR,
            'embeddings': cls.EMBEDDINGS_DIR,
            'attendance': cls.ATTENDANCE_DIR,
            'unknown_faces': cls.UNKNOWN_FACES_DIR,
            'models': cls.MODELS_DIR,
            'logs': cls.LOGS_DIR,
            'notebooks': cls.NOTEBOOKS_DIR
        }


# Create a singleton instance
config = Config()

# Export commonly used constants
FACE_DETECTION_CONFIDENCE = config.FACE_DETECTION_CONFIDENCE
FACE_RECOGNITION_THRESHOLD = config.FACE_RECOGNITION_THRESHOLD
IMAGE_SIZE = config.IMAGE_SIZE
IMAGES_PER_STUDENT = config.IMAGES_PER_STUDENT