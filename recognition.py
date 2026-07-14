"""
Real-time face recognition module.

This module handles real-time face detection and recognition using
MediaPipe for detection and FaceNet/InsightFace for recognition.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from config import config
from database import db
from student import student_manager
from embedding_generator import embedding_generator
from utils import Utils

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Real-time face recognition class."""
    
    def __init__(self):
        """Initialize the face recognition system."""
        # Initialize face detector with fallback
        self.face_detector = self._initialize_face_detector()
        
        self.utils = Utils()
        self.embedding_generator = embedding_generator
        self.threshold = config.FACE_RECOGNITION_THRESHOLD
        
        # Load embeddings
        self.embeddings = self.embedding_generator.load_all_embeddings()
        self.student_info = self._load_student_info()
        
        # Initialize model for recognition
        self.model = embedding_generator.model
        self.model_type = embedding_generator.model_type
        
        # Tracking variables
        self.tracked_faces = {}
        self.tracking_timeout = 2.0  # seconds
        self.last_attendance_time = {}
        
        logger.info("FaceRecognizer initialized successfully")
    
    def _initialize_face_detector(self):
        """
        Initialize face detector with fallback options.
        
        Returns:
            Face detector object
        """
        # Try MediaPipe first
        try:
            import mediapipe as mp
            
            # Check if solutions is available (older versions)
            if hasattr(mp, 'solutions'):
                self.mp_face_detection = mp.solutions.face_detection
                detector = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
                )
                logger.info("Using MediaPipe face detection (solutions API)")
                return detector
            
            # Try newer MediaPipe API (tasks)
            elif hasattr(mp, 'tasks'):
                from mediapipe.tasks import python
                from mediapipe.tasks.python.vision import face_detector
                
                base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
                options = face_detector.FaceDetectorOptions(
                    base_options=base_options,
                    running_mode=face_detector.RunningMode.IMAGE,
                    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
                )
                detector = face_detector.FaceDetector.create_from_options(options)
                logger.info("Using MediaPipe face detection (tasks API)")
                return detector
            
            else:
                logger.warning("MediaPipe installed but no detection API found")
                raise ImportError("MediaPipe detection API not found")
                
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
            logger.info("Falling back to OpenCV Haar Cascade...")
            return self._initialize_opencv_detector()
    
    def _initialize_opencv_detector(self):
        """
        Initialize OpenCV Haar Cascade face detector as fallback.
        
        Returns:
            OpenCV CascadeClassifier
        """
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            try:
                detector = cv2.CascadeClassifier(path)
                if not detector.empty():
                    logger.info(f"Using OpenCV Haar Cascade from: {path}")
                    return detector
            except:
                continue
        
        raise RuntimeError("No face detector available. Please install MediaPipe or OpenCV.")
    
    def _load_student_info(self) -> Dict[str, Dict]:
        """
        Load student information for recognition.
        
        Returns:
            Dictionary mapping roll_number to student info
        """
        students = student_manager.get_all_students()
        info = {}
        for student in students:
            roll = student.get('roll_number')
            if roll:
                info[roll] = {
                    'name': student.get('name', 'Unknown'),
                    'student_id': str(student.get('_id', '')),
                    'department': student.get('department', ''),
                    'section': student.get('section', '')
                }
        return info
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the frame using available detector.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected face information
        """
        try:
            # Check which detector we're using
            if hasattr(self.face_detector, 'process'):
                # MediaPipe detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb_frame)
                
                faces = []
                if results.detections:
                    h, w, _ = frame.shape
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Add padding
                        padding = 20
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        width = min(w - x, width + 2 * padding)
                        height = min(h - y, height + 2 * padding)
                        
                        if width > 0 and height > 0:
                            faces.append({
                                'bbox': (x, y, width, height),
                                'confidence': detection.score[0]
                            })
                
                return faces
            
            else:
                # OpenCV Haar Cascade detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_detected = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                faces = []
                for (x, y, w, h) in faces_detected:
                    # Add padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2 * padding)
                    h = min(frame.shape[0] - y, h + 2 * padding)
                    
                    if w > 0 and h > 0:
                        faces.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.9  # Default confidence for Haar
                        })
                
                return faces
                
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract and preprocess face from frame.
        
        Args:
            frame: Input image frame
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Preprocessed face image or None
        """
        try:
            x, y, w, h = bbox
            
            # Extract face
            face = frame[y:y+h, x:x+w]
            
            if face.size == 0:
                return None
            
            # Resize to target size
            target_size = config.IMAGE_SIZE
            face = cv2.resize(face, target_size)
            
            # Convert to RGB
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            return face_rgb
            
        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None
    
    def recognize_face(self, face: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Recognize a face by comparing embeddings.
        
        Args:
            face: Preprocessed face image
            
        Returns:
            Tuple of (roll_number, name, confidence)
        """
        try:
            # Generate embedding for the face
            embedding = self.embedding_generator.generate_embedding(face)
            
            if embedding is None:
                return None, None, 0.0
            
            # If no embeddings to compare against
            if not self.embeddings:
                return None, None, 0.0
            
            # Compare against stored embeddings
            best_roll = None
            best_name = None
            best_similarity = 0.0
            
            for roll, stored_emb in self.embeddings.items():
                # Calculate cosine similarity
                similarity = cosine_similarity([embedding], [stored_emb])[0][0]
                
                if similarity > best_similarity and similarity >= self.threshold:
                    best_similarity = similarity
                    best_roll = roll
                    if roll in self.student_info:
                        best_name = self.student_info[roll].get('name', 'Unknown')
            
            return best_roll, best_name, best_similarity
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None, None, 0.0
    
    def process_frame(self, frame: np.ndarray, capture_unknown: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame for face recognition.
        
        Args:
            frame: Input image frame
            capture_unknown: Whether to save unknown faces
            
        Returns:
            Tuple of (annotated_frame, recognized_faces)
        """
        try:
            # Mirror frame for natural display
            display_frame = cv2.flip(frame, 1)
            recognized_faces = []
            
            # Detect faces
            faces = self.detect_faces(display_frame)
            
            for face_info in faces:
                bbox = face_info['bbox']
                detection_confidence = face_info['confidence']
                
                # Extract face
                face_image = self.extract_face(display_frame, bbox)
                
                if face_image is None:
                    continue
                
                # Recognize face
                roll_number, name, similarity = self.recognize_face(face_image)
                
                x, y, w, h = bbox
                
                if roll_number and name:
                    # Recognized face
                    color = (0, 255, 0)  # Green
                    label = f"{name} ({roll_number})"
                    confidence = f"{similarity*100:.1f}%"
                    
                    recognized_faces.append({
                        'roll_number': roll_number,
                        'name': name,
                        'bbox': bbox,
                        'confidence': similarity,
                        'status': 'recognized'
                    })
                else:
                    # Unknown face
                    color = (0, 0, 255)  # Red
                    label = "Unknown"
                    confidence = ""
                    
                    # Save unknown face if requested
                    if capture_unknown and face_image is not None:
                        self._save_unknown_face(face_image)
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label and confidence
                cv2.putText(display_frame, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if confidence:
                    cv2.putText(display_frame, confidence, (x, y+h+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return display_frame, recognized_faces
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def _save_unknown_face(self, face_image: np.ndarray) -> None:
        """
        Save unknown face for later review.
        
        Args:
            face_image: Face image to save
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}.jpg"
            filepath = config.UNKNOWN_FACES_DIR / filename
            
            # Convert to BGR for saving
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), face_bgr)
            
            logger.info(f"Saved unknown face: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving unknown face: {e}")
    
    def run_realtime_recognition(self, capture_unknown: bool = False) -> None:
        """
        Run real-time face recognition from webcam.
        
        Args:
            capture_unknown: Whether to save unknown faces
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info("Starting real-time face recognition. Press 'q' to quit, 's' to save unknown faces.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                display_frame, recognized_faces = self.process_frame(frame, capture_unknown)
                
                # Display info
                cv2.putText(display_frame, f"Faces: {len(recognized_faces)}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'q' to quit, 's' to toggle save unknown", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Real-Time Face Recognition', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    capture_unknown = not capture_unknown
                    status = "ON" if capture_unknown else "OFF"
                    logger.info(f"Unknown face saving: {status}")
        
        except Exception as e:
            logger.error(f"Error during recognition: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def update_embeddings(self) -> None:
        """Reload embeddings and student information."""
        self.embeddings = self.embedding_generator.load_all_embeddings()
        self.student_info = self._load_student_info()
        logger.info("Embeddings and student info updated")


# Create a singleton instance
recognizer = FaceRecognizer()


if __name__ == "__main__":
    # Run real-time recognition
    print("Starting Real-Time Face Recognition")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Toggle saving unknown faces")
    print("=" * 50)
    
    recognizer.run_realtime_recognition()