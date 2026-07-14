from pymongo import MongoClient

try:
    client = MongoClient('mongodb://localhost:27017/')
    client.admin.command('ping')
    print("✅ MongoDB connection successful!")
    client.close()
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")