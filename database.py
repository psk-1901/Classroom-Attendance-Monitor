"""
Database module for MongoDB operations.

This module handles all database interactions including connections,
CRUD operations, and collection management.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from pymongo.collection import Collection
from pymongo.database import Database

from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB database handler for the attendance system."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string. If None, uses config.
        """
        self.connection_string = connection_string or config.MONGODB_URI
        self.database_name = config.DATABASE_NAME
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"Successfully connected to MongoDB database: {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection from the database.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection object
        """
        return self.db[collection_name]
    
    # ==================== STUDENT OPERATIONS ====================
    
    def register_student(self, student_data: Dict[str, Any]) -> str:
        """
        Register a new student in the database.
        
        Args:
            student_data: Dictionary containing student information
            
        Returns:
            Student ID of the registered student
            
        Raises:
            DuplicateKeyError: If roll number already exists
        """
        collection = self.get_collection(config.COLLECTIONS['students'])
        
        # Check if roll number already exists
        existing = collection.find_one({'roll_number': student_data['roll_number']})
        if existing:
            raise DuplicateKeyError(f"Student with roll number {student_data['roll_number']} already exists")
        
        # Add timestamps
        student_data['created_at'] = datetime.now()
        student_data['updated_at'] = datetime.now()
        student_data['face_captured'] = False
        student_data['embeddings_generated'] = False
        
        try:
            result = collection.insert_one(student_data)
            logger.info(f"Student registered successfully: {student_data['name']} ({student_data['roll_number']})")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to register student: {e}")
            raise
    
    def get_student_by_roll_number(self, roll_number: str) -> Optional[Dict[str, Any]]:
        """
        Get student information by roll number.
        
        Args:
            roll_number: Student's roll number
            
        Returns:
            Student document or None if not found
        """
        collection = self.get_collection(config.COLLECTIONS['students'])
        return collection.find_one({'roll_number': roll_number})
    
    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get student information by ID.
        
        Args:
            student_id: Student's MongoDB ObjectId
            
        Returns:
            Student document or None if not found
        """
        from bson import ObjectId
        collection = self.get_collection(config.COLLECTIONS['students'])
        try:
            return collection.find_one({'_id': ObjectId(student_id)})
        except:
            return None
    
    def get_all_students(self) -> List[Dict[str, Any]]:
        """
        Get all registered students.
        
        Returns:
            List of student documents
        """
        collection = self.get_collection(config.COLLECTIONS['students'])
        return list(collection.find())
    
    def update_student(self, roll_number: str, update_data: Dict[str, Any]) -> bool:
        """
        Update student information.
        
        Args:
            roll_number: Student's roll number
            update_data: Dictionary with fields to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        collection = self.get_collection(config.COLLECTIONS['students'])
        update_data['updated_at'] = datetime.now()
        
        result = collection.update_one(
            {'roll_number': roll_number},
            {'$set': update_data}
        )
        
        if result.modified_count > 0:
            logger.info(f"Student updated: {roll_number}")
            return True
        return False
    
    # ==================== EMBEDDING OPERATIONS ====================
    
    def store_embedding(self, embedding_data: Dict[str, Any]) -> str:
        """
        Store face embedding for a student.
        
        Args:
            embedding_data: Dictionary containing embedding information
            
        Returns:
            Embedding document ID
        """
        collection = self.get_collection(config.COLLECTIONS['embeddings'])
        
        # Check if embedding already exists for this student
        existing = collection.find_one({'student_id': embedding_data['student_id']})
        if existing:
            # Update existing embedding
            embedding_data['updated_at'] = datetime.now()
            result = collection.update_one(
                {'student_id': embedding_data['student_id']},
                {'$set': embedding_data}
            )
            logger.info(f"Embedding updated for student: {embedding_data['student_id']}")
            return str(result.upserted_id)
        else:
            # Insert new embedding
            embedding_data['created_at'] = datetime.now()
            embedding_data['updated_at'] = datetime.now()
            result = collection.insert_one(embedding_data)
            logger.info(f"Embedding stored for student: {embedding_data['student_id']}")
            return str(result.inserted_id)
    
    def get_embedding(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get face embedding for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            Embedding document or None if not found
        """
        collection = self.get_collection(config.COLLECTIONS['embeddings'])
        return collection.find_one({'student_id': student_id})
    
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get all face embeddings.
        
        Returns:
            List of embedding documents
        """
        collection = self.get_collection(config.COLLECTIONS['embeddings'])
        return list(collection.find())
    
    # ==================== ATTENDANCE OPERATIONS ====================
    
    def mark_attendance(self, attendance_data: Dict[str, Any]) -> str:
        """
        Mark attendance for a student.
        
        Args:
            attendance_data: Dictionary containing attendance information
            
        Returns:
            Attendance document ID
        """
        collection = self.get_collection(config.COLLECTIONS['attendance'])
        
        # Check if attendance already marked for this session and student
        existing = collection.find_one({
            'session_id': attendance_data['session_id'],
            'student_id': attendance_data['student_id']
        })
        
        if existing:
            logger.warning(f"Attendance already marked for student: {attendance_data['student_id']}")
            return str(existing['_id'])
        
        attendance_data['created_at'] = datetime.now()
        result = collection.insert_one(attendance_data)
        logger.info(f"Attendance marked for student: {attendance_data['student_id']}")
        return str(result.inserted_id)
    
    def get_attendance_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all attendance records for a session.
        
        Args:
            session_id: Attendance session ID
            
        Returns:
            List of attendance records
        """
        collection = self.get_collection(config.COLLECTIONS['attendance'])
        return list(collection.find({'session_id': session_id}))
    
    def get_attendance_by_student(self, roll_number: str) -> List[Dict[str, Any]]:
        """
        Get attendance history for a student.
        
        Args:
            roll_number: Student's roll number
            
        Returns:
            List of attendance records
        """
        collection = self.get_collection(config.COLLECTIONS['attendance'])
        return list(collection.find({'roll_number': roll_number}))
    
    def get_attendance_by_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Get attendance records for a specific date.
        
        Args:
            date: Date string in format YYYY-MM-DD
            
        Returns:
            List of attendance records
        """
        collection = self.get_collection(config.COLLECTIONS['attendance'])
        return list(collection.find({'date': date}))
    
    # ==================== LOG OPERATIONS ====================
    
    def log_event(self, event_data: Dict[str, Any]) -> str:
        """
        Log an event in the system.
        
        Args:
            event_data: Dictionary containing event information
            
        Returns:
            Log document ID
        """
        collection = self.get_collection(config.COLLECTIONS['logs'])
        event_data['timestamp'] = datetime.now()
        result = collection.insert_one(event_data)
        return str(result.inserted_id)
    
    # ==================== UTILITY OPERATIONS ====================
    
    def delete_student(self, roll_number: str) -> bool:
        """
        Delete a student and their associated data.
        
        Args:
            roll_number: Student's roll number
            
        Returns:
            True if deleted successfully, False otherwise
        """
        student = self.get_student_by_roll_number(roll_number)
        if not student:
            return False
        
        student_id = str(student['_id'])
        
        # Delete student
        collection_students = self.get_collection(config.COLLECTIONS['students'])
        collection_students.delete_one({'roll_number': roll_number})
        
        # Delete embeddings
        collection_embeddings = self.get_collection(config.COLLECTIONS['embeddings'])
        collection_embeddings.delete_one({'student_id': student_id})
        
        # Delete attendance records
        collection_attendance = self.get_collection(config.COLLECTIONS['attendance'])
        collection_attendance.delete_many({'student_id': student_id})
        
        logger.info(f"Student deleted: {roll_number}")
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with collection counts
        """
        return {
            'students': self.get_collection(config.COLLECTIONS['students']).count_documents({}),
            'embeddings': self.get_collection(config.COLLECTIONS['embeddings']).count_documents({}),
            'attendance': self.get_collection(config.COLLECTIONS['attendance']).count_documents({}),
            'logs': self.get_collection(config.COLLECTIONS['logs']).count_documents({})
        }
    
    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Create a singleton instance
db = MongoDB()


if __name__ == "__main__":
    # Test the database connection
    try:
        stats = db.get_stats()
        print("Database Statistics:")
        for collection, count in stats.items():
            print(f"  {collection}: {count}")
        db.close()
    except Exception as e:
        print(f"Error: {e}")