"""
Main entry point for the attendance system.

This module provides a command-line interface for running different
components of the attendance system.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import config
from database import db
from student import student_manager
from capture_faces import face_capture
from embedding_generator import embedding_generator
from recognition import recognizer
from attendance import attendance_manager
from excel_report import excel_report
from dashboard import dashboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def register_student():
    """Register a new student."""
    print("\n=== Student Registration ===\n")
    
    # Get student details
    student_data = {
        'student_id': input("Student ID: ").strip(),
        'roll_number': input("Roll Number: ").strip().upper(),
        'name': input("Student Name: ").strip(),
        'department': input("Department: ").strip(),
        'section': input("Section: ").strip(),
        'year': input("Year (1st/2nd/3rd/4th/5th): ").strip(),
        'email': input("Email: ").strip(),
        'phone': input("Phone Number: ").strip()
    }
    
    # Register student
    success, message, student_id = student_manager.register_student(student_data)
    
    if success:
        print(f"\n✓ {message}")
        print(f"Student ID: {student_id}")
    else:
        print(f"\n✗ {message}")


def capture_faces():
    """Capture face images for students."""
    print("\n=== Face Capture ===\n")
    
    # Show summary
    summary = face_capture.get_capture_summary()
    print(f"Total Students: {summary['total_students']}")
    print(f"Faces Captured: {summary['captured']}")
    print(f"Pending: {summary['pending']}")
    print()
    
    if summary['pending'] == 0:
        print("All students have faces captured.")
        return
    
    # Ask for specific student or all
    choice = input("Enter roll number to capture (or 'all' for all pending): ").strip()
    
    if choice.lower() == 'all':
        results = face_capture.capture_all_students()
        print(f"\nCapture Results:")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        
        for detail in results['details']:
            print(f"  {detail['name']}: {detail['status']}")
    
    else:
        success, message, count = face_capture.capture_faces_for_student(choice)
        if success:
            print(f"\n✓ {message}")
        else:
            print(f"\n✗ {message}")


def generate_embeddings():
    """Generate face embeddings for students."""
    print("\n=== Generate Embeddings ===\n")
    
    # Show summary
    summary = embedding_generator.get_embedding_summary()
    print(f"Total Students: {summary['total_students']}")
    print(f"Faces Captured: {summary['face_captured']}")
    print(f"Embeddings Generated: {summary['embeddings_generated']}")
    print(f"Pending: {summary['pending']}")
    print()
    
    if summary['pending'] == 0:
        print("All students have embeddings generated.")
        return
    
    # Ask for specific student or all
    choice = input("Enter roll number to generate (or 'all' for all pending): ").strip()
    
    if choice.lower() == 'all':
        results = embedding_generator.generate_all_embeddings()
        print(f"\nGeneration Results:")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        
        for detail in results['details']:
            print(f"  {detail['name']}: {detail['status']}")
    
    else:
        success, message, embedding = embedding_generator.generate_embeddings_for_student(choice)
        if success:
            print(f"\n✓ {message}")
            if embedding is not None:
                print(f"Embedding shape: {embedding.shape}")
        else:
            print(f"\n✗ {message}")


def run_recognition():
    """Run real-time face recognition."""
    print("\n=== Real-Time Face Recognition ===\n")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Toggle saving unknown faces")
    print("  'u' - Update embeddings")
    print()
    
    # Check if embeddings exist
    embeddings = embedding_generator.load_all_embeddings()
    if not embeddings:
        print("No embeddings found. Please generate embeddings first.")
        return
    
    print(f"Loaded {len(embeddings)} embeddings for recognition")
    print("Starting recognition...")
    print()
    
    recognizer.run_realtime_recognition()


def run_attendance():
    """Run attendance marking session."""
    print("\n=== Mark Attendance ===\n")
    
    # Start session
    subject = input("Subject: ").strip()
    faculty = input("Faculty: ").strip()
    classroom = input("Classroom (optional): ").strip()
    
    session_id = attendance_manager.start_attendance_session(subject, faculty, classroom)
    print(f"\nSession started: {session_id}")
    print("Press 'q' to finish attendance\n")
    
    # Run recognition
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    recognized_faces = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process frame
            display_frame, faces = recognizer.process_frame(frame)
            
            # Collect recognized faces
            for face in faces:
                if face['status'] == 'recognized':
                    # Avoid duplicates
                    roll = face['roll_number']
                    if not any(f['roll_number'] == roll for f in recognized_faces):
                        recognized_faces.append(face)
            
            # Show status
            cv2.putText(display_frame, f"Recognized: {len(recognized_faces)}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to finish attendance", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Attendance Session', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Mark attendance
    print(f"\nFinishing attendance session...")
    print(f"Recognized {len(recognized_faces)} unique students")
    
    summary = attendance_manager.mark_attendance_for_session(session_id, recognized_faces)
    
    print(f"\nAttendance Summary:")
    print(f"  Total Students: {summary['total_students']}")
    print(f"  Present: {summary['present']}")
    print(f"  Absent: {summary['absent']}")
    print(f"  Attendance: {summary['attendance_percentage']:.1f}%")
    print(f"\nSession completed: {session_id}")


def generate_report():
    """Generate attendance report."""
    print("\n=== Generate Attendance Report ===\n")
    
    session_id = input("Enter session ID: ").strip()
    
    if not session_id:
        print("Session ID required")
        return
    
    try:
        filepath = excel_report.generate_attendance_report(session_id)
        print(f"\n✓ Report generated successfully")
        print(f"  Location: {filepath}")
    except Exception as e:
        print(f"\n✗ Error generating report: {e}")


def run_dashboard():
    """Run the dashboard."""
    print("\n=== Dashboard ===\n")
    print("Starting dashboard...")
    print("Press 'q' to quit")
    
    dashboard.run_dashboard()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Attendance System')
    parser.add_argument('command', 
                       choices=['register', 'capture', 'embeddings', 'recognize', 
                               'attendance', 'report', 'dashboard'],
                       help='Command to execute')
    
    args = parser.parse_args()
    
    # Execute command
    commands = {
        'register': register_student,
        'capture': capture_faces,
        'embeddings': generate_embeddings,
        'recognize': run_recognition,
        'attendance': run_attendance,
        'report': generate_report,
        'dashboard': run_dashboard
    }
    
    try:
        commands[args.command]()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode
        print("\n=== Attendance System ===\n")
        print("Commands:")
        print("  1. Register Student")
        print("  2. Capture Faces")
        print("  3. Generate Embeddings")
        print("  4. Real-Time Recognition")
        print("  5. Mark Attendance")
        print("  6. Generate Report")
        print("  7. Dashboard")
        print("  8. Exit")
        print()
        
        while True:
            choice = input("Enter choice (1-8): ").strip()
            
            if choice == '1':
                register_student()
            elif choice == '2':
                capture_faces()
            elif choice == '3':
                generate_embeddings()
            elif choice == '4':
                run_recognition()
            elif choice == '5':
                run_attendance()
            elif choice == '6':
                generate_report()
            elif choice == '7':
                run_dashboard()
            elif choice == '8':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
            
            print("\n" + "-" * 50 + "\n")