# 📸 Classroom Attendance System

A deep learning-based classroom attendance system using real-time face recognition with MediaPipe and FaceNet.

## 🎯 Features

- ✅ Student Registration with MongoDB
- ✅ Face Dataset Collection (100 images/student)
- ✅ FaceNet Embedding Generation (128-dim vectors)
- ✅ Real-time Face Recognition with Cosine Similarity
- ✅ Smart Attendance Marking (Auto-mark absent students)
- ✅ Professional Excel Reports with Color Coding
- ✅ Interactive Dashboard with Live Camera Feed
- ✅ Jupyter Notebooks for Each Module

## 🛠️ Tech Stack

- **Python 3.11+**
- **MediaPipe** - Face Detection
- **FaceNet / InsightFace** - Face Recognition
- **MongoDB** - Database
- **OpenCV** - Image Processing
- **TensorFlow** - Deep Learning
- **Jupyter Notebook** - Interactive Development

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/attendance-system.git
cd attendance-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install MongoDB
```bash
# Windows
net start MongoDB
# Linux
sudo systemctl start mongod
# macOS
brew services start mongodb-community
```

### 5. Configure Environment
Create `.env` file:
```env
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=attendance_system
LOG_LEVEL=INFO
```

## 🚀 Quick Start

### Using Jupyter Notebooks (Recommended)
```bash
jupyter notebook
```
Run notebooks in order: **01 → 07**

### Using Command Line
```bash
# Register student
python main.py register

# Capture faces
python main.py capture

# Generate embeddings
python main.py embeddings

# Take attendance
python main.py attendance

# Generate report
python main.py report

# Launch dashboard
python main.py dashboard
```

### Interactive Mode
```bash
python main.py
```

## 📁 Project Structure

```
attendance-system/
├── notebooks/               # Jupyter Notebooks (01-07)
├── attendance_system/       # Python modules
│   ├── config.py
│   ├── database.py
│   ├── student.py
│   ├── capture_faces.py
│   ├── embedding_generator.py
│   ├── recognition.py
│   ├── attendance.py
│   ├── excel_report.py
│   ├── dashboard.py
│   └── main.py
├── dataset/                 # Captured face images
├── embeddings/              # Face embeddings
├── Attendance/              # Generated reports
├── requirements.txt
└── README.md
```

## 📊 Workflow

1. **Register Students** → Add student details to MongoDB
2. **Capture Faces** → Collect 100 face images per student
3. **Generate Embeddings** → Create FaceNet embeddings
4. **Take Attendance** → Real-time face recognition
5. **Generate Reports** → Export Excel attendance reports

## 🔧 Troubleshooting

### MongoDB Connection Failed
```bash
# Windows
net start MongoDB
# Linux
sudo systemctl start mongod
# macOS
brew services start mongodb-community
```

### Webcam Not Working
```python
import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found")
```

### MediaPipe Issues
```bash
pip install --upgrade mediapipe
# System automatically falls back to OpenCV Haar Cascade
```

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/)
- [FaceNet](https://github.com/davidsandberg/facenet)
- [TensorFlow](https://www.tensorflow.org/)
- [MongoDB](https://www.mongodb.com/)

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/attendance-system/issues)
- **Email**: sanjaykrishnapati@gmail.com, sanjay1409kumar@gmail.com

---

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the education community
