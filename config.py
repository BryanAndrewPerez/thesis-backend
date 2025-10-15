import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Firebase Configuration
    FIREBASE_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL', 'https://air-quality-monitoring-f97b2-default-rtdb.asia-southeast1.firebasedatabase.app/')
    FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.getenv(FIREBASE_SERVICE_ACCOUNT_KEY_PATH = 'path/to/serviceAccountKey.json')
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Model paths
    MODELS_DIR = 'models'
    
    # API Configuration
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:3001']
