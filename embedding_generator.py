"""
Face embedding generation module.

This module handles generating FaceNet embeddings for all captured faces
and storing them for recognition.
"""

import os
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import cv2
from sklearn.preprocessing import normalize

from config import config
from database import db
from student import student_manager
from utils import Utils

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Face embedding generation class using FaceNet."""
    
    def __init__(self):
        """Initialize the embedding generator."""
        self.utils = Utils()
        self.model = None
        self.model_type = None
        self.embeddings_file = config.EMBEDDINGS_FILE
        
        # Try loading FaceNet, fallback to InsightFace
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the face recognition model."""
        try:
            # Try to import and load FaceNet
            from keras_facenet import FaceNet
            self.model = FaceNet()
            self.model_type = 'facenet'
            logger.info("FaceNet model loaded successfully")
        except ImportError as e:
            logger.warning(f"FaceNet not available: {e}. Falling back to InsightFace.")
            self._load_insightface()
        except Exception as e:
            logger.warning(f"Error loading FaceNet: {e}. Falling back to InsightFace.")
            self._load_insightface()
    
    def _load_insightface(self) -> None:
        """Load InsightFace as fallback."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
            
            self.model = FaceAnalysis(name='buffalo_l')
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            self.model_type = 'insightface'
            logger.info("InsightFace model loaded successfully")
        except ImportError as e:
            logger.error(f"InsightFace not available: {e}")
            raise RuntimeError("Neither FaceNet nor InsightFace could be loaded")
        except Exception as e:
            logger.error(f"Error loading InsightFace: {e}")
            raise RuntimeError(f"Failed to load face recognition models: {e}")
    
    def generate_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding for a face image.
        
        Args:
            image: Face image (160x160)
            
        Returns:
            128-dimensional embedding vector or None if failed
        """
        try:
            if self.model_type == 'facenet':
                # FaceNet expects batch of images
                if len(image.shape) == 3:
                    image = np.expand_dims(image, axis=0)
                embedding = self.model.embeddings(image)
                return embedding[0] if embedding is not None else None
            
            elif self.model_type == 'insightface':
                # InsightFace expects face detection first
                faces = self.model.get(image)
                if len(faces) > 0:
                    embedding = faces[0].normed_embedding
                    return embedding
                return None
            
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_for_student(self, roll_number: str) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Generate embeddings for a specific student's face images.
        
        Args:
            roll_number: Student's roll number
            
        Returns:
            Tuple of (success, message, average_embedding)
        """
        try:
            # Get student info
            student = student_manager.get_student(roll_number)
            if not student:
                return False, f"Student {roll_number} not found", None
            
            # Get student directory
            student_dir = config.DATASET_DIR / student['name']
            if not student_dir.exists():
                return False, f"No face images found for {student['name']}", None
            
            # Get all face images
            images = list(student_dir.glob('*.jpg'))
            if not images:
                return False, f"No face images found for {student['name']}", None
            
            # Generate embeddings for each image
            embeddings = []
            
            with tqdm(total=len(images), desc=f"Generating embeddings for {student['name']}") as pbar:
                for img_path in images:
                    try:
                        # Read image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Generate embedding
                        embedding = self.generate_embedding(img_rgb)
                        if embedding is not None:
                            embeddings.append(embedding)
                    except Exception as e:
                        logger.warning(f"Error processing {img_path}: {e}")
                    
                    pbar.update(1)
            
            if not embeddings:
                return False, "No valid embeddings generated", None
            
            # Calculate average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Normalize embedding
            avg_embedding = normalize([avg_embedding])[0]
            
            # Store embedding
            embedding_data = {
                'student_id': str(student['_id']),
                'roll_number': roll_number,
                'name': student['name'],
                'embedding': avg_embedding.tolist(),
                'num_images': len(embeddings),
                'model_type': self.model_type
            }
            
            # Store in MongoDB
            db.store_embedding(embedding_data)
            
            # Update student record
            student_manager.update_student(roll_number, {'embeddings_generated': True})
            
            # Also save to local file
            self._save_embedding_locally(roll_number, avg_embedding, student)
            
            return True, f"Embeddings generated for {student['name']}", avg_embedding
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return False, f"Error: {str(e)}", None
    
    def _save_embedding_locally(self, roll_number: str, embedding: np.ndarray, student: Dict) -> None:
        """
        Save embedding to local pickle file.
        
        Args:
            roll_number: Student's roll number
            embedding: Embedding vector
            student: Student information
        """
        try:
            # Load existing embeddings
            embeddings_data = self._load_embeddings_from_file()
            
            # Update or add
            embeddings_data[roll_number] = {
                'name': student['name'],
                'roll_number': roll_number,
                'embedding': embedding.tolist(),
                'student_id': str(student['_id'])
            }
            
            # Save to file
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            logger.info(f"Embedding saved locally for {student['name']}")
            
        except Exception as e:
            logger.error(f"Error saving embedding locally: {e}")
    
    def _load_embeddings_from_file(self) -> Dict:
        """
        Load embeddings from local pickle file.
        
        Returns:
            Dictionary of embeddings
        """
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading embeddings file: {e}")
                return {}
        return {}
    
    def generate_all_embeddings(self) -> Dict[str, Any]:
        """
        Generate embeddings for all students with captured faces.
        
        Returns:
            Dictionary with generation results
        """
        students = student_manager.get_all_students()
        results = {
            'total': len(students),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for student in students:
            if not student.get('face_captured', False):
                results['details'].append({
                    'name': student['name'],
                    'roll_number': student['roll_number'],
                    'status': 'Skipped (No face captured)'
                })
                continue
            
            if student.get('embeddings_generated', False):
                results['details'].append({
                    'name': student['name'],
                    'roll_number': student['roll_number'],
                    'status': 'Skipped (Already generated)'
                })
                continue
            
            success, message, _ = self.generate_embeddings_for_student(student['roll_number'])
            
            if success:
                results['successful'] += 1
                status = 'Success'
            else:
                results['failed'] += 1
                status = f'Failed: {message}'
            
            results['details'].append({
                'name': student['name'],
                'roll_number': student['roll_number'],
                'status': status
            })
        
        return results
    
    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load all embeddings from database or local file.
        
        Returns:
            Dictionary mapping roll_number to embedding
        """
        try:
            # Try loading from local file first
            if self.embeddings_file.exists():
                data = self._load_embeddings_from_file()
                embeddings = {}
                for roll, info in data.items():
                    if 'embedding' in info:
                        embeddings[roll] = np.array(info['embedding'])
                
                if embeddings:
                    logger.info(f"Loaded {len(embeddings)} embeddings from local file")
                    return embeddings
            
            # If local file doesn't exist or is empty, load from database
            embedding_docs = db.get_all_embeddings()
            embeddings = {}
            
            for doc in embedding_docs:
                roll = doc.get('roll_number')
                emb = doc.get('embedding')
                if roll and emb:
                    embeddings[roll] = np.array(emb)
            
            logger.info(f"Loaded {len(embeddings)} embeddings from database")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {}
    
    def get_embedding_summary(self) -> Dict[str, Any]:
        """
        Get summary of embedding generation status.
        
        Returns:
            Dictionary with summary information
        """
        students = student_manager.get_all_students()
        
        summary = {
            'total_students': len(students),
            'face_captured': 0,
            'embeddings_generated': 0,
            'pending': 0
        }
        
        for student in students:
            if student.get('face_captured', False):
                summary['face_captured'] += 1
                if student.get('embeddings_generated', False):
                    summary['embeddings_generated'] += 1
                else:
                    summary['pending'] += 1
        
        return summary


# Create a singleton instance
embedding_generator = EmbeddingGenerator()


if __name__ == "__main__":
    # Test embedding generation
    print("Embedding Generation System")
    print("=" * 50)
    
    # Get summary
    summary = embedding_generator.get_embedding_summary()
    print(f"Total students: {summary['total_students']}")
    print(f"Face captured: {summary['face_captured']}")
    print(f"Embeddings generated: {summary['embeddings_generated']}")
    print(f"Pending: {summary['pending']}")
    
    # Generate embeddings for a specific student
    roll_number = input("Enter roll number to generate embeddings (or press Enter to skip): ")
    if roll_number:
        success, message, embedding = embedding_generator.generate_embeddings_for_student(roll_number)
        print(f"Result: {success} - {message}")
        if success and embedding is not None:
            print(f"Embedding shape: {embedding.shape}")