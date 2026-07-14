"""
Face dataset collection module.

This module handles capturing face images from webcam, processing them,
and saving them to the dataset directory.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import logging
from datetime import datetime
import time
from tqdm import tqdm
from PIL import Image

from config import config
from database import db
from student import student_manager
from utils import Utils

logger = logging.getLogger(__name__)


class FaceCapture:
    """Face capture and dataset collection class."""
    
    def __init__(self):
        """Initialize the face capture system."""
        self.utils = Utils()
        self.dataset_dir = config.DATASET_DIR
        
        # Create dataset directory if it doesn't exist
        self.dataset_dir.mkdir(exist_ok=True)
        
        # Initialize face detector
        self.face_detector = self._initialize_face_detector()
        
        logger.info("FaceCapture initialized successfully")
    
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
                
                # Create detector
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
        # Try to load Haar Cascade
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
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, tuple]]:
        """
        Detect face in frame using available detector.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (cropped_face, bounding_box) or None if no face detected
        """
        try:
            # Check which detector we're using
            if hasattr(self.face_detector, 'process'):
                # MediaPipe detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb_frame)
                
                if not results.detections:
                    return None
                
                # Get the first face
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                h, w, _ = frame.shape
                
                # Calculate absolute coordinates
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
                
                # Crop face
                face = frame[y:y+height, x:x+width]
                
                if face.size == 0:
                    return None
                
                return face, (x, y, width, height)
            
            else:
                # OpenCV Haar Cascade detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) == 0:
                    return None
                
                # Get the first face
                x, y, w, h = faces[0]
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                # Crop face
                face = frame[y:y+h, x:x+w]
                
                if face.size == 0:
                    return None
                
                return face, (x, y, w, h)
                
        except Exception as e:
            logger.error(f"Error detecting face: {e}")
            return None
    
    def is_blurry(self, image: np.ndarray, threshold: float = None) -> bool:
        """
        Check if image is blurry using variance of Laplacian.
        
        Args:
            image: Input image
            threshold: Blur threshold (lower = more blur detection)
            
        Returns:
            True if image is blurry, False otherwise
        """
        if threshold is None:
            threshold = config.BLUR_THRESHOLD
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    
    def preprocess_face(self, face: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess face image: resize, normalize, and enhance.
        
        Args:
            face: Input face image
            
        Returns:
            Preprocessed face image or None if invalid
        """
        try:
            if face.size == 0:
                return None
            
            # Resize to target size
            target_size = config.IMAGE_SIZE
            resized = cv2.resize(face, target_size)
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Convert back to uint8 for saving
            preprocessed = (normalized * 255).astype(np.uint8)
            
            return preprocessed
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None
    
    def remove_background(self, face: np.ndarray) -> np.ndarray:
        """
        Remove background using simple thresholding.
        
        Args:
            face: Input face image
            
        Returns:
            Face image with background removed
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            
            # Apply mask
            result = cv2.bitwise_and(face, face, mask=mask)
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return face
    
    def capture_faces_for_student(self, roll_number: str, num_images: int = None) -> Tuple[bool, str, int]:
        """
        Capture face images for a specific student.
        
        Args:
            roll_number: Student's roll number
            num_images: Number of images to capture
            
        Returns:
            Tuple of (success, message, captured_count)
        """
        if num_images is None:
            num_images = config.IMAGES_PER_STUDENT
        
        # Get student info
        student = student_manager.get_student(roll_number)
        if not student:
            return False, f"Student with roll number {roll_number} not found", 0
        
        # Create student directory
        student_dir = self.dataset_dir / student['name']
        student_dir.mkdir(exist_ok=True)
        
        # Check existing images
        existing_images = list(student_dir.glob('*.jpg'))
        start_index = len(existing_images)
        
        if start_index >= num_images:
            return True, f"Already captured {num_images} images for {student['name']}", num_images
        
        remaining = num_images - start_index
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Could not open webcam. Please check camera connection.", 0
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        captured = start_index
        last_capture_time = time.time()
        capture_interval = config.CAPTURE_INTERVAL
        
        logger.info(f"Starting face capture for {student['name']}. Need {remaining} more images.")
        
        try:
            # Create progress bar
            pbar = tqdm(total=remaining, desc=f"Capturing {student['name']}", unit="img")
            
            while captured < num_images:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                # Mirror frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Detect face
                result = self.detect_face(frame)
                
                # Create display frame
                display_frame = frame.copy()
                
                if result:
                    face, bbox = result
                    
                    # Check if face is blurry
                    if self.is_blurry(face):
                        cv2.putText(display_frame, "Blurry face - adjust position", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # Preprocess face
                        processed_face = self.preprocess_face(face)
                        
                        if processed_face is not None:
                            # Save image if enough time has passed
                            current_time = time.time()
                            if current_time - last_capture_time >= capture_interval:
                                # Remove background
                                processed_face = self.remove_background(processed_face)
                                
                                # Save image
                                img_path = student_dir / f"{captured+1:04d}.jpg"
                                cv2.imwrite(str(img_path), cv2.cvtColor(processed_face, cv2.COLOR_RGB2BGR))
                                
                                captured += 1
                                last_capture_time = current_time
                                pbar.update(1)
                                
                                # Update progress
                                progress = (captured / num_images) * 100
                                cv2.putText(display_frame, f"Progress: {progress:.1f}% ({captured}/{num_images})", 
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw bounding box
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw face preview
                    preview_h, preview_w = 100, 100
                    cv2.rectangle(display_frame, (10, display_frame.shape[0]-preview_h-10), 
                                (10+preview_w, display_frame.shape[0]-10), (255, 255, 255), -1)
                    if face.size > 0:
                        face_preview = cv2.resize(face, (preview_w, preview_h))
                        display_frame[display_frame.shape[0]-preview_h-10:display_frame.shape[0]-10, 
                                    10:10+preview_w] = face_preview
                
                else:
                    cv2.putText(display_frame, "No face detected - please show your face", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show instructions
                cv2.putText(display_frame, f"Student: {student['name']} ({student['roll_number']})", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Images: {captured}/{num_images} | Press 'q' to quit", 
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Face Capture - Press "q" to quit', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except Exception as e:
            logger.error(f"Error during face capture: {e}")
            return False, f"Error during face capture: {str(e)}", captured
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            pbar.close()
        
        # Update student record
        if captured >= num_images:
            student_manager.update_student(roll_number, {'face_captured': True})
            return True, f"Face capture completed for {student['name']}", captured
        else:
            return False, f"Face capture incomplete. Captured {captured}/{num_images} images", captured
    
    def capture_all_students(self, force_retake: bool = False) -> Dict[str, Any]:
        """
        Capture face images for all students without captured faces.
        
        Args:
            force_retake: If True, recapture for all students
            
        Returns:
            Dictionary with capture results
        """
        students = student_manager.get_all_students()
        results = {
            'total': len(students),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for student in students:
            if student.get('face_captured', False) and not force_retake:
                results['details'].append({
                    'name': student['name'],
                    'roll_number': student['roll_number'],
                    'status': 'Skipped (Already captured)'
                })
                continue
            
            success, message, count = self.capture_faces_for_student(student['roll_number'])
            
            if success:
                results['successful'] += 1
                status = f"Completed ({count} images)"
            else:
                results['failed'] += 1
                status = f"Failed: {message}"
            
            results['details'].append({
                'name': student['name'],
                'roll_number': student['roll_number'],
                'status': status
            })
        
        return results
    
    def get_capture_summary(self) -> Dict[str, Any]:
        """
        Get summary of face capture status.
        
        Returns:
            Dictionary with capture summary
        """
        students = student_manager.get_all_students()
        
        summary = {
            'total_students': len(students),
            'captured': 0,
            'pending': 0,
            'students_with_faces': []
        }
        
        for student in students:
            if student.get('face_captured', False):
                summary['captured'] += 1
                summary['students_with_faces'].append({
                    'name': student['name'],
                    'roll_number': student['roll_number']
                })
            else:
                summary['pending'] += 1
        
        return summary


# Create a singleton instance
face_capture = FaceCapture()


if __name__ == "__main__":
    # Test face capture
    print("Face Capture System")
    print("=" * 50)
    
    # Get summary
    summary = face_capture.get_capture_summary()
    print(f"Total students: {summary['total_students']}")
    print(f"Face captured: {summary['captured']}")
    print(f"Pending: {summary['pending']}")
    
    # Capture for a specific student
    roll_number = input("Enter roll number to capture faces (or press Enter to skip): ")
    if roll_number:
        success, message, count = face_capture.capture_faces_for_student(roll_number)
        print(f"Result: {success} - {message} ({count} images)")