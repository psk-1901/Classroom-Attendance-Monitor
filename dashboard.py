"""
Dashboard module for the attendance system.

This module provides a visual dashboard with statistics, live camera feed,
and system monitoring capabilities.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from PIL import Image

from config import config
from database import db
from student import student_manager
from attendance import attendance_manager
from recognition import recognizer

logger = logging.getLogger(__name__)


class Dashboard:
    """Dashboard class for system monitoring and statistics."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.db = db
        self.student_manager = student_manager
        self.attendance_manager = attendance_manager
        self.recognizer = recognizer
        
        self.stats = {}
        self.attendance_data = {}
        self.last_update = None
    
    def update_stats(self) -> Dict[str, Any]:
        """
        Update all dashboard statistics.
        
        Returns:
            Dictionary with updated statistics
        """
        try:
            # Get student statistics
            student_count = self.student_manager.get_student_count()
            capture_status = self.student_manager.get_face_capture_status()
            
            # Get attendance statistics for today
            today = datetime.now().strftime('%Y-%m-%d')
            today_attendance = self.attendance_manager.get_attendance_summary(date=today)
            
            # Get embedding status
            embedding_summary = self.recognizer.embedding_generator.get_embedding_summary()
            
            # Get database statistics
            db_stats = self.db.get_stats()
            
            self.stats = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'students': {
                    'total': student_count,
                    'face_captured': capture_status.get('face_captured', 0),
                    'embeddings_generated': capture_status.get('embeddings_generated', 0),
                    'pending_capture': capture_status.get('pending_capture', 0),
                    'pending_embeddings': capture_status.get('pending_embeddings', 0)
                },
                'attendance': {
                    'total': today_attendance.get('total', 0),
                    'present': today_attendance.get('present', 0),
                    'absent': today_attendance.get('absent', 0),
                    'percentage': today_attendance.get('attendance_percentage', 0.0)
                },
                'database': db_stats,
                'embeddings': embedding_summary
            }
            
            self.last_update = datetime.now()
            return self.stats
            
        except Exception as e:
            logger.error(f"Error updating dashboard stats: {e}")
            return {}
    
    def get_attendance_chart_data(self, days: int = 7) -> Dict[str, Any]:
        """
        Get attendance data for chart visualization.
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary with chart data
        """
        try:
            # Get attendance records for the last N days
            dates = []
            present_counts = []
            absent_counts = []
            percentages = []
            
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                summary = self.attendance_manager.get_attendance_summary(date=date)
                
                dates.append(date)
                present_counts.append(summary.get('present', 0))
                absent_counts.append(summary.get('absent', 0))
                
                total = summary.get('total', 0)
                if total > 0:
                    percentages.append((summary.get('present', 0) / total) * 100)
                else:
                    percentages.append(0.0)
            
            # Reverse to show chronological order
            dates.reverse()
            present_counts.reverse()
            absent_counts.reverse()
            percentages.reverse()
            
            return {
                'dates': dates,
                'present': present_counts,
                'absent': absent_counts,
                'percentages': percentages
            }
            
        except Exception as e:
            logger.error(f"Error getting chart data: {e}")
            return {}
    
    def generate_dashboard_frame(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a dashboard frame with statistics and optional camera feed.
        
        Args:
            frame: Optional camera frame to include
            
        Returns:
            Dashboard frame as numpy array
        """
        try:
            # Update stats
            stats = self.update_stats()
            
            # Create a blank frame for the dashboard
            height = 720
            width = 1280
            dashboard = np.zeros((height, width, 3), dtype=np.uint8)
            dashboard.fill(30)  # Dark background
            
            # Title
            cv2.putText(dashboard, "ATTENDANCE SYSTEM DASHBOARD", 
                      (width//2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                      1.0, (255, 255, 255), 2)
            
            # Statistics boxes
            x_offset = 20
            y_offset = 80
            box_width = 280
            box_height = 120
            margin = 15
            
            # Box 1: Students
            self._draw_stat_box(dashboard, x_offset, y_offset, box_width, box_height,
                              "STUDENTS", 
                              f"Total: {stats.get('students', {}).get('total', 0)}",
                              f"Faces Captured: {stats.get('students', {}).get('face_captured', 0)}",
                              f"Embeddings: {stats.get('students', {}).get('embeddings_generated', 0)}")
            
            # Box 2: Today's Attendance
            x_offset += box_width + margin
            self._draw_stat_box(dashboard, x_offset, y_offset, box_width, box_height,
                              "TODAY'S ATTENDANCE",
                              f"Present: {stats.get('attendance', {}).get('present', 0)}",
                              f"Absent: {stats.get('attendance', {}).get('absent', 0)}",
                              f"Attendance: {stats.get('attendance', {}).get('percentage', 0):.1f}%")
            
            # Box 3: Database
            x_offset += box_width + margin
            self._draw_stat_box(dashboard, x_offset, y_offset, box_width, box_height,
                              "DATABASE",
                              f"Students: {stats.get('database', {}).get('students', 0)}",
                              f"Attendance: {stats.get('database', {}).get('attendance', 0)}",
                              f"Embeddings: {stats.get('database', {}).get('embeddings', 0)}")
            
            # Box 4: System Status
            x_offset += box_width + margin
            self._draw_stat_box(dashboard, x_offset, y_offset, box_width, box_height,
                              "SYSTEM STATUS",
                              f"Last Update: {self.last_update.strftime('%H:%M:%S') if self.last_update else 'N/A'}",
                              f"Model: {self.recognizer.model_type.upper()}",
                              f"Threshold: {self.recognizer.threshold:.2f}")
            
            # Camera feed (if provided)
            if frame is not None:
                camera_height = 360
                camera_width = 640
                y_offset = y_offset + box_height + margin
                
                # Resize frame
                cam_display = cv2.resize(frame, (camera_width, camera_height))
                
                # Place camera feed on dashboard
                dashboard[y_offset:y_offset+camera_height, 
                        width//2 - camera_width//2:width//2 + camera_width//2] = cam_display
                
                # Label
                cv2.putText(dashboard, "LIVE CAMERA FEED", 
                          (width//2 - 100, y_offset - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating dashboard frame: {e}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def _draw_stat_box(self, frame: np.ndarray, x: int, y: int, width: int, height: int,
                      title: str, *lines) -> None:
        """
        Draw a statistics box on the frame.
        
        Args:
            frame: Frame to draw on
            x, y: Position
            width, height: Box dimensions
            title: Box title
            *lines: Text lines to display
        """
        # Draw background
        cv2.rectangle(frame, (x, y), (x+width, y+height), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, y), (x+width, y+height), (100, 100, 100), 1)
        
        # Title
        cv2.putText(frame, title, (x+10, y+25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
        
        # Lines
        for i, line in enumerate(lines):
            y_pos = y + 45 + (i * 22)
            cv2.putText(frame, line, (x+10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_dashboard(self) -> None:
        """
        Run the dashboard in real-time with camera feed.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info("Starting dashboard. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Generate dashboard
                dashboard = self.generate_dashboard_frame(frame)
                
                # Display
                cv2.imshow('Attendance System Dashboard', dashboard)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


# Create a singleton instance
dashboard = Dashboard()


if __name__ == "__main__":
    # Run dashboard
    print("Starting Dashboard")
    print("=" * 50)
    print("Press 'q' to quit")
    
    dashboard.run_dashboard()