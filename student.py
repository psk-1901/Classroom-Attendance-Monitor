"""
Student management module.

This module handles all student-related operations including registration,
validation, and data management.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from bson import ObjectId
import re

from database import db
from config import config

logger = logging.getLogger(__name__)


class StudentManager:
    """Manager class for student operations."""
    
    def __init__(self):
        """Initialize the student manager."""
        self.db = db
    
    def validate_student_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate student data before registration.
        
        Args:
            data: Student data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Required fields
        required_fields = ['student_id', 'roll_number', 'name', 'department', 
                          'section', 'year', 'email', 'phone']
        
        for field in required_fields:
            if field not in data or not data[field]:
                return False, f"Missing required field: {field}"
        
        # Validate roll number format (adjust pattern as needed)
        if not re.match(r'^[A-Z0-9]{5,15}$', data['roll_number']):
            return False, "Invalid roll number format"
        
        # Validate email
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data['email']):
            return False, "Invalid email format"
        
        # Validate phone number (10 digits)
        if not re.match(r'^\d{10}$', data['phone']):
            return False, "Phone number must be 10 digits"
        
        # Validate year
        valid_years = ['1st', '2nd', '3rd', '4th', '5th']
        if data['year'] not in valid_years:
            return False, f"Year must be one of: {', '.join(valid_years)}"
        
        return True, ""
    
    def register_student(self, data: Dict[str, Any]) -> tuple[bool, str, Optional[str]]:
        """
        Register a new student.
        
        Args:
            data: Student data dictionary
            
        Returns:
            Tuple of (success, message, student_id)
        """
        try:
            # Validate data
            is_valid, error = self.validate_student_data(data)
            if not is_valid:
                return False, error, None
            
            # Check for existing roll number
            existing = self.db.get_student_by_roll_number(data['roll_number'])
            if existing:
                return False, f"Student with roll number {data['roll_number']} already exists", None
            
            # Register student
            student_id = self.db.register_student(data)
            return True, "Student registered successfully", student_id
            
        except Exception as e:
            logger.error(f"Error registering student: {e}")
            return False, f"Error registering student: {str(e)}", None
    
    def get_student(self, roll_number: str) -> Optional[Dict[str, Any]]:
        """
        Get student information by roll number.
        
        Args:
            roll_number: Student's roll number
            
        Returns:
            Student document or None if not found
        """
        try:
            return self.db.get_student_by_roll_number(roll_number)
        except Exception as e:
            logger.error(f"Error getting student: {e}")
            return None
    
    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get student information by ID.
        
        Args:
            student_id: Student's MongoDB ObjectId
            
        Returns:
            Student document or None if not found
        """
        try:
            return self.db.get_student_by_id(student_id)
        except Exception as e:
            logger.error(f"Error getting student by ID: {e}")
            return None
    
    def get_all_students(self) -> List[Dict[str, Any]]:
        """
        Get all registered students.
        
        Returns:
            List of student documents
        """
        try:
            return self.db.get_all_students()
        except Exception as e:
            logger.error(f"Error getting all students: {e}")
            return []
    
    def update_student(self, roll_number: str, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Update student information.
        
        Args:
            roll_number: Student's roll number
            data: Updated data dictionary
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Remove sensitive fields that shouldn't be updated
            data.pop('_id', None)
            data.pop('created_at', None)
            data.pop('roll_number', None)  # Don't allow roll number change
            
            success = self.db.update_student(roll_number, data)
            if success:
                return True, "Student updated successfully"
            return False, "Student not found"
            
        except Exception as e:
            logger.error(f"Error updating student: {e}")
            return False, f"Error updating student: {str(e)}"
    
    def delete_student(self, roll_number: str) -> tuple[bool, str]:
        """
        Delete a student and their associated data.
        
        Args:
            roll_number: Student's roll number
            
        Returns:
            Tuple of (success, message)
        """
        try:
            success = self.db.delete_student(roll_number)
            if success:
                return True, "Student deleted successfully"
            return False, "Student not found"
            
        except Exception as e:
            logger.error(f"Error deleting student: {e}")
            return False, f"Error deleting student: {str(e)}"
    
    def get_students_by_department(self, department: str) -> List[Dict[str, Any]]:
        """
        Get students by department.
        
        Args:
            department: Department name
            
        Returns:
            List of student documents
        """
        try:
            collection = self.db.get_collection(config.COLLECTIONS['students'])
            return list(collection.find({'department': department}))
        except Exception as e:
            logger.error(f"Error getting students by department: {e}")
            return []
    
    def get_students_by_section(self, section: str) -> List[Dict[str, Any]]:
        """
        Get students by section.
        
        Args:
            section: Section name
            
        Returns:
            List of student documents
        """
        try:
            collection = self.db.get_collection(config.COLLECTIONS['students'])
            return list(collection.find({'section': section}))
        except Exception as e:
            logger.error(f"Error getting students by section: {e}")
            return []
    
    def get_students_by_year(self, year: str) -> List[Dict[str, Any]]:
        """
        Get students by year.
        
        Args:
            year: Academic year
            
        Returns:
            List of student documents
        """
        try:
            collection = self.db.get_collection(config.COLLECTIONS['students'])
            return list(collection.find({'year': year}))
        except Exception as e:
            logger.error(f"Error getting students by year: {e}")
            return []
    
    def get_student_count(self) -> int:
        """
        Get total number of registered students.
        
        Returns:
            Number of students
        """
        try:
            collection = self.db.get_collection(config.COLLECTIONS['students'])
            return collection.count_documents({})
        except Exception as e:
            logger.error(f"Error getting student count: {e}")
            return 0
    
    def get_face_capture_status(self) -> Dict[str, Any]:
        """
        Get face capture status for all students.
        
        Returns:
            Dictionary with capture statistics
        """
        try:
            all_students = self.get_all_students()
            total = len(all_students)
            captured = sum(1 for s in all_students if s.get('face_captured', False))
            embeddings = sum(1 for s in all_students if s.get('embeddings_generated', False))
            
            return {
                'total_students': total,
                'face_captured': captured,
                'embeddings_generated': embeddings,
                'pending_capture': total - captured,
                'pending_embeddings': captured - embeddings
            }
        except Exception as e:
            logger.error(f"Error getting face capture status: {e}")
            return {}


# Create a singleton instance
student_manager = StudentManager()


if __name__ == "__main__":
    # Test student management
    test_student = {
        'student_id': 'STU001',
        'roll_number': '22001',
        'name': 'Test Student',
        'department': 'Computer Science',
        'section': 'A',
        'year': '3rd',
        'email': 'test@example.com',
        'phone': '1234567890'
    }
    
    success, message, student_id = student_manager.register_student(test_student)
    print(f"Registration: {success}, {message}")
    
    if success:
        student = student_manager.get_student('22001')
        print(f"Retrieved student: {student}")