"""
Attendance management module.

This module handles all attendance operations including marking attendance,
generating reports, and managing attendance sessions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid
import pandas as pd

from config import config
from database import db
from student import student_manager
from recognition import recognizer

logger = logging.getLogger(__name__)


class AttendanceManager:
    """Attendance management class."""
    
    def __init__(self):
        """Initialize the attendance manager."""
        self.db = db
        self.student_manager = student_manager
        self.recognizer = recognizer
        self.current_session = None
        self.session_start_time = None
    
    def start_attendance_session(self, subject: str, faculty: str, classroom: str = "") -> str:
        """
        Start a new attendance session.
        
        Args:
            subject: Subject name
            faculty: Faculty name
            classroom: Classroom name
            
        Returns:
            Session ID
        """
        session_id = f"ATT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        session_data = {
            'session_id': session_id,
            'subject': subject,
            'faculty': faculty,
            'classroom': classroom,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'start_time': datetime.now().strftime('%H:%M:%S'),
            'status': 'active'
        }
        
        # Store session info in local and database
        self.current_session = session_data
        self.session_start_time = datetime.now()
        
        # Log to database
        db.log_event({
            'event_type': 'attendance_session_start',
            'session_id': session_id,
            'details': session_data
        })
        
        logger.info(f"Attendance session started: {session_id}")
        return session_id
    
    def mark_attendance_for_session(self, session_id: str, recognized_faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mark attendance for all students based on recognized faces.
        
        This is the most important method. It uses the student database as the master list
        and marks all recognized students as present, all others as absent.
        
        Args:
            session_id: Attendance session ID
            recognized_faces: List of recognized face information
            
        Returns:
            Attendance summary statistics
        """
        try:
            # Get session info
            session_info = self._get_session_info(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")
            
            # Get all students
            all_students = self.student_manager.get_all_students()
            
            if not all_students:
                logger.warning("No students registered in the system")
                return {
                    'total_students': 0,
                    'present': 0,
                    'absent': 0,
                    'attendance_percentage': 0.0
                }
            
            # Convert recognized_faces list to a dictionary for easy lookup
            recognized_dict = {}
            for face in recognized_faces:
                roll = face.get('roll_number')
                if roll:
                    recognized_dict[roll] = face
            
            # Extract recognized roll numbers
            recognized_rolls = set(recognized_dict.keys())
            
            # Prepare attendance records
            attendance_records = []
            present_count = 0
            absent_count = 0
            
            current_time = datetime.now().strftime('%H:%M:%S')
            date = datetime.now().strftime('%Y-%m-%d')
            
            for student in all_students:
                roll = student.get('roll_number')
                name = student.get('name', 'Unknown')
                student_id = str(student.get('_id', ''))
                
                # Determine status
                if roll in recognized_rolls:
                    status = 'present'
                    time_in = current_time
                    # Get confidence from recognized_dict
                    face_info = recognized_dict.get(roll, {})
                    confidence = face_info.get('confidence', 0.0)
                    present_count += 1
                else:
                    status = 'absent'
                    time_in = '--'
                    confidence = 0.0
                    absent_count += 1
                
                # Create attendance record
                record = {
                    'session_id': session_id,
                    'student_id': student_id,
                    'roll_number': roll,
                    'name': name,
                    'department': student.get('department', ''),
                    'section': student.get('section', ''),
                    'year': student.get('year', ''),
                    'date': date,
                    'time_in': time_in,
                    'status': status,
                    'confidence': confidence,
                    'subject': session_info.get('subject', ''),
                    'faculty': session_info.get('faculty', ''),
                    'classroom': session_info.get('classroom', '')
                }
                
                # Store in database (only if not already marked)
                existing = self.db.get_collection(config.COLLECTIONS['attendance']).find_one({
                    'session_id': session_id,
                    'student_id': student_id
                })
                
                if not existing:
                    db.mark_attendance(record)
                
                attendance_records.append(record)
            
            # Update session status
            self._update_session_status(session_id, 'completed')
            
            # Log completion
            db.log_event({
                'event_type': 'attendance_session_complete',
                'session_id': session_id,
                'statistics': {
                    'total': len(all_students),
                    'present': present_count,
                    'absent': absent_count,
                    'attendance_percentage': (present_count / len(all_students) * 100) if all_students else 0
                }
            })
            
            logger.info(f"Attendance session {session_id} completed. Present: {present_count}, Absent: {absent_count}")
            
            return {
                'total_students': len(all_students),
                'present': present_count,
                'absent': absent_count,
                'attendance_percentage': (present_count / len(all_students) * 100) if all_students else 0,
                'records': attendance_records
            }
            
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            raise
    
    def _get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get attendance session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information dictionary
        """
        # Check current session
        if self.current_session and self.current_session.get('session_id') == session_id:
            return self.current_session
        
        # Check database
        logs = self.db.get_collection(config.COLLECTIONS['logs'])
        session_log = logs.find_one({
            'event_type': 'attendance_session_start',
            'session_id': session_id
        })
        
        if session_log:
            return session_log.get('details', {})
        
        return None
    
    def _update_session_status(self, session_id: str, status: str) -> None:
        """
        Update session status.
        
        Args:
            session_id: Session ID
            status: New status
        """
        if self.current_session and self.current_session.get('session_id') == session_id:
            self.current_session['status'] = status
        
        # Update in database
        logs = self.db.get_collection(config.COLLECTIONS['logs'])
        logs.update_one(
            {'session_id': session_id, 'event_type': 'attendance_session_start'},
            {'$set': {'details.status': status}}
        )
    
    def get_attendance_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all attendance records for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of attendance records
        """
        return self.db.get_attendance_by_session(session_id)
    
    def get_attendance_by_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Get attendance records for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            List of attendance records
        """
        collection = self.db.get_collection(config.COLLECTIONS['attendance'])
        return list(collection.find({'date': date}))
    
    def get_attendance_by_student(self, roll_number: str) -> List[Dict[str, Any]]:
        """
        Get attendance history for a student.
        
        Args:
            roll_number: Student's roll number
            
        Returns:
            List of attendance records
        """
        return self.db.get_attendance_by_student(roll_number)
    
    def get_attendance_summary(self, session_id: str = None, date: str = None) -> Dict[str, Any]:
        """
        Get attendance summary for a session or date.
        
        Args:
            session_id: Session ID
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Attendance summary
        """
        if session_id:
            records = self.get_attendance_by_session(session_id)
        elif date:
            records = self.get_attendance_by_date(date)
        else:
            # Get today's attendance
            date = datetime.now().strftime('%Y-%m-%d')
            records = self.get_attendance_by_date(date)
        
        if not records:
            return {
                'total': 0,
                'present': 0,
                'absent': 0,
                'attendance_percentage': 0.0
            }
        
        total = len(records)
        present = sum(1 for r in records if r.get('status') == 'present')
        absent = total - present
        
        return {
            'total': total,
            'present': present,
            'absent': absent,
            'attendance_percentage': (present / total * 100) if total > 0 else 0.0,
            'records': records
        }
    
    def get_attendance_by_filters(self, **filters) -> List[Dict[str, Any]]:
        """
        Get attendance records with filters.
        
        Args:
            **filters: Filter criteria (date, roll_number, subject, faculty, etc.)
            
        Returns:
            List of filtered attendance records
        """
        try:
            collection = self.db.get_collection(config.COLLECTIONS['attendance'])
            query = {}
            
            for key, value in filters.items():
                if value and value != 'all':
                    query[key] = value
            
            return list(collection.find(query))
            
        except Exception as e:
            logger.error(f"Error getting filtered attendance: {e}")
            return []
    
    def get_attendance_statistics(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Get comprehensive attendance statistics.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.db.get_collection(config.COLLECTIONS['attendance'])
            
            query = {}
            if start_date and end_date:
                query['date'] = {'$gte': start_date, '$lte': end_date}
            elif start_date:
                query['date'] = {'$gte': start_date}
            elif end_date:
                query['date'] = {'$lte': end_date}
            
            records = list(collection.find(query))
            
            if not records:
                return {
                    'total_sessions': 0,
                    'total_records': 0,
                    'overall_present': 0,
                    'overall_absent': 0,
                    'overall_attendance_percentage': 0.0,
                    'student_stats': []
                }
            
            # Group by student
            student_stats = {}
            for record in records:
                roll = record.get('roll_number')
                if roll not in student_stats:
                    student_stats[roll] = {
                        'roll_number': roll,
                        'name': record.get('name', ''),
                        'department': record.get('department', ''),
                        'total_sessions': 0,
                        'present': 0,
                        'absent': 0
                    }
                
                student_stats[roll]['total_sessions'] += 1
                if record.get('status') == 'present':
                    student_stats[roll]['present'] += 1
                else:
                    student_stats[roll]['absent'] += 1
            
            # Calculate statistics
            total_present = sum(s['present'] for s in student_stats.values())
            total_absent = sum(s['absent'] for s in student_stats.values())
            total_records = total_present + total_absent
            
            # Add attendance percentage
            for stats in student_stats.values():
                total = stats['total_sessions']
                if total > 0:
                    stats['attendance_percentage'] = (stats['present'] / total) * 100
                else:
                    stats['attendance_percentage'] = 0.0
            
            return {
                'total_sessions': len(set(r.get('session_id') for r in records)),
                'total_records': total_records,
                'overall_present': total_present,
                'overall_absent': total_absent,
                'overall_attendance_percentage': (total_present / total_records * 100) if total_records > 0 else 0.0,
                'student_stats': list(student_stats.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting attendance statistics: {e}")
            return {}


# Create a singleton instance
attendance_manager = AttendanceManager()


if __name__ == "__main__":
    # Test attendance management
    print("Attendance Management System")
    print("=" * 50)
    
    # Start a session
    session_id = attendance_manager.start_attendance_session(
        subject="Machine Learning",
        faculty="Dr. John Doe",
        classroom="Room 201"
    )
    print(f"Session started: {session_id}")
    
    # Get summary
    summary = attendance_manager.get_attendance_summary(session_id)
    print(f"Attendance Summary: {summary}")