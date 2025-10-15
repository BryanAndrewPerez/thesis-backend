import firebase_admin
from firebase_admin import credentials, db
import os
from datetime import datetime, timedelta
import json

class FirebaseService:
    def __init__(self, service_account_path, database_url):
        self.service_account_path = service_account_path
        self.database_url = database_url
        self.initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.service_account_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': self.database_url
                })
            self.initialized = True
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Firebase initialization error: {e}")
            self.initialized = False
    
    def fetch_sensor_data(self, location="", hours=24):
        """Fetch sensor data from Firebase for the last N hours"""
        if not self.initialized:
            return None
            
        try:
            # Get current time and calculate start time
            now = datetime.now()
            start_time = now - timedelta(hours=hours)
            
            # Fetch data from Firebase - using 'sensors' collection
            ref = db.reference('/sensors')
            if location:
                ref = ref.child(location)
                
            data = ref.get()
            
            if not data:
                return None
                
            # Process the data to get the last 24 hours
            processed_data = []
            
            for timestamp, readings in data.items():
                try:
                    # Parse timestamp
                    data_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    
                    # Only include data from the specified time range
                    if data_time >= start_time:
                        processed_data.append({
                            'timestamp': timestamp,
                            'time': data_time,
                            'pm25': readings.get('pm25', 0),
                            'pm10': readings.get('pm10', 0),
                            'no2': readings.get('no2', 0),
                            'co': readings.get('co', 0),
                            'so2': readings.get('so2', 0)
                        })
                except Exception as e:
                    print(f"Error processing timestamp {timestamp}: {e}")
                    continue
                    
            # Sort by timestamp and get last N hours
            processed_data.sort(key=lambda x: x['time'])
            return processed_data[-hours:] if len(processed_data) >= hours else processed_data
            
        except Exception as e:
            print(f"Error fetching Firebase data: {e}")
            return None
    
    def get_available_locations(self):
        """Get list of available sensor locations"""
        if not self.initialized:
            return []
            
        try:
            ref = db.reference('/sensors')
            data = ref.get()
            
            if not data:
                return []
                
            # Get all location keys
            locations = list(data.keys())
            return locations
            
        except Exception as e:
            print(f"Error fetching locations: {e}")
            return []
    
    def get_latest_reading(self, location=""):
        """Get the latest sensor reading"""
        if not self.initialized:
            return None
            
        try:
            ref = db.reference('/sensors')
            if location:
                ref = ref.child(location)
                
            data = ref.order_by_key().limit_to_last(1).get()
            
            if not data:
                return None
                
            # Get the latest reading
            latest_timestamp = max(data.keys())
            latest_reading = data[latest_timestamp]
            
            return {
                'timestamp': latest_timestamp,
                'time': datetime.strptime(latest_timestamp, "%Y-%m-%d %H:%M:%S"),
                'pm25': latest_reading.get('pm25', 0),
                'pm10': latest_reading.get('pm10', 0),
                'no2': latest_reading.get('no2', 0),
                'co': latest_reading.get('co', 0),
                'so2': latest_reading.get('so2', 0)
            }
            
        except Exception as e:
            print(f"Error fetching latest reading: {e}")
            return None
