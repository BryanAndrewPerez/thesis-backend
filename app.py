from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import pickle
import os
import firebase_admin
from firebase_admin import credentials, db
import json
from datetime import datetime, timedelta
import math
from config import Config


app = Flask(__name__)
CORS(app, origins=['https://lipaircast.onrender.com'], methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], allow_headers=['Content-Type', 'Authorization'])


# Initialize Firebase Admin SDK

def initialize_firebase():
    try:
        print("🔍 Initializing Firebase...")
        print(f"🔍 FIREBASE_DATABASE_URL: {Config.FIREBASE_DATABASE_URL}")
        print(f"🔍 FIREBASE_SERVICE_ACCOUNT_KEY exists: {Config.FIREBASE_SERVICE_ACCOUNT_KEY is not None}")
        
        # Check if environment variables are set
        import os
        print(f"🔍 Environment FIREBASE_DATABASE_URL: {os.getenv('FIREBASE_DATABASE_URL')}")
        print(f"🔍 Environment FIREBASE_SERVICE_ACCOUNT_KEY exists: {os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY') is not None}")
        
        if not firebase_admin._apps:
            # Use service account from config
            service_account_info = Config.FIREBASE_SERVICE_ACCOUNT_KEY
            if not service_account_info:
                print("❌ FIREBASE_SERVICE_ACCOUNT_KEY is None or empty")
                return False
                
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred, {
                'databaseURL': Config.FIREBASE_DATABASE_URL
            })
            print("✅ Firebase initialized successfully")
        else:
            print("✅ Firebase already initialized")
        return True
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

# === Load PM model and scalers ===
PM_MODEL_PATH = "models/newchunk_pmonly.keras"
PM_INPUT_SCALER_PATH = "models/newinput_scaler_pmonly.pkl"
PM_TARGET_SCALERS_PATH = "models/newtarget_scalers_pmonly.pkl"

pm_model = None
pm_input_scaler = None
pm_target_scalers = None

# === Load NO2 model and scalers ===
NO2_MODEL_PATH = "models/chunk_no2only.keras"
NO2_INPUT_SCALER_PATH = "models/input_scaler_no2only.pkl"
NO2_TARGET_SCALERS_PATH = "models/target_scalers_no2only.pkl"

no2_model = None
no2_input_scaler = None
no2_target_scalers = None

# === Load CO model and scalers ===
CO_MODEL_PATH = "models/chunk_coonly (1).keras"
CO_INPUT_SCALER_PATH = "models/input_scaler_coonly.pkl"
CO_TARGET_SCALERS_PATH = "models/target_scalers_coonly.pkl"

co_model = None
co_input_scaler = None
co_target_scalers = None

def load_models():
    """Load all ML models and scalers"""
    global pm_model, pm_input_scaler, pm_target_scalers
    global no2_model, no2_input_scaler, no2_target_scalers
    global co_model, co_input_scaler, co_target_scalers
    
    print("🔍 Starting model loading process...")
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Models directory exists: {os.path.exists('models')}")
    print(f"🔍 TensorFlow version: {tf.__version__}")
    
    # List all files in models directory
    if os.path.exists('models'):
        model_files = os.listdir('models')
        print(f"🔍 Files in models directory: {model_files}")
    
    try:
        # Load PM model and scalers
        print(f"🔍 Checking PM model at: {PM_MODEL_PATH}")
        if os.path.exists(PM_MODEL_PATH):
            print("✅ PM model file found, loading...")
            try:
                pm_model = tf.keras.models.load_model(PM_MODEL_PATH, compile=False)
                print(f"✅ PM model loaded successfully, shape: {pm_model.input_shape}")
            except Exception as e:
                print(f"❌ PM model loading failed: {e}")
                import traceback
                traceback.print_exc()
                pm_model = None
            
            try:
                with open(PM_INPUT_SCALER_PATH, "rb") as f:
                    pm_input_scaler = pickle.load(f)
                print("✅ PM input scaler loaded")
            except Exception as e:
                print(f"❌ PM input scaler loading failed: {e}")
                pm_input_scaler = None
                
            try:
                with open(PM_TARGET_SCALERS_PATH, "rb") as f:
                    pm_target_scalers = pickle.load(f)
                print("✅ PM target scalers loaded")
            except Exception as e:
                print(f"❌ PM target scalers loading failed: {e}")
                pm_target_scalers = None
        else:
            print(f"❌ PM model not found at {PM_MODEL_PATH}")
            
        # Load NO2 model and scalers
        print(f"🔍 Checking NO2 model at: {NO2_MODEL_PATH}")
        if os.path.exists(NO2_MODEL_PATH):
            print("✅ NO2 model file found, loading...")
            try:
                no2_model = tf.keras.models.load_model(NO2_MODEL_PATH, compile=False)
                print(f"✅ NO2 model loaded successfully, shape: {no2_model.input_shape}")
            except Exception as e:
                print(f"❌ NO2 model loading failed: {e}")
                import traceback
                traceback.print_exc()
                no2_model = None
            
            try:
                with open(NO2_INPUT_SCALER_PATH, "rb") as f:
                    no2_input_scaler = pickle.load(f)
                print("✅ NO2 input scaler loaded")
            except Exception as e:
                print(f"❌ NO2 input scaler loading failed: {e}")
                no2_input_scaler = None
                
            try:
                with open(NO2_TARGET_SCALERS_PATH, "rb") as f:
                    no2_target_scalers = pickle.load(f)
                print("✅ NO2 target scalers loaded")
            except Exception as e:
                print(f"❌ NO2 target scalers loading failed: {e}")
                no2_target_scalers = None
        else:
            print(f"❌ NO2 model not found at {NO2_MODEL_PATH}")
            
        # Load CO model and scalers
        print(f"🔍 Checking CO model at: {CO_MODEL_PATH}")
        if os.path.exists(CO_MODEL_PATH):
            print("✅ CO model file found, loading...")
            try:
                co_model = tf.keras.models.load_model(CO_MODEL_PATH, compile=False)
                print(f"✅ CO model loaded successfully, shape: {co_model.input_shape}")
            except Exception as e:
                print(f"❌ CO model loading failed: {e}")
                import traceback
                traceback.print_exc()
                co_model = None
            
            try:
                with open(CO_INPUT_SCALER_PATH, "rb") as f:
                    co_input_scaler = pickle.load(f)
                print("✅ CO input scaler loaded")
            except Exception as e:
                print(f"❌ CO input scaler loading failed: {e}")
                co_input_scaler = None
                
            try:
                with open(CO_TARGET_SCALERS_PATH, "rb") as f:
                    co_target_scalers = pickle.load(f)
                print("✅ CO target scalers loaded")
            except Exception as e:
                print(f"❌ CO target scalers loading failed: {e}")
                co_target_scalers = None
        else:
            print(f"❌ CO model not found at {CO_MODEL_PATH}")
            
        # Summary
        print("📊 Model Loading Summary:")
        print(f"   PM Model: {'✅ Loaded' if pm_model is not None else '❌ Failed'}")
        print(f"   NO2 Model: {'✅ Loaded' if no2_model is not None else '❌ Failed'}")
        print(f"   CO Model: {'✅ Loaded' if co_model is not None else '❌ Failed'}")
        print(f"   PM Scalers: {'✅ Loaded' if pm_input_scaler is not None and pm_target_scalers is not None else '❌ Failed'}")
        print(f"   NO2 Scalers: {'✅ Loaded' if no2_input_scaler is not None and no2_target_scalers is not None else '❌ Failed'}")
        print(f"   CO Scalers: {'✅ Loaded' if co_input_scaler is not None and co_target_scalers is not None else '❌ Failed'}")
            
    except Exception as e:
        print(f"❌ Critical error loading models: {e}")
        import traceback
        traceback.print_exc()

def get_model_files_status():
    """Return whether each model file and scaler exists on disk."""
    return {
        "pm": {
            "model_exists": os.path.exists(PM_MODEL_PATH),
            "input_scaler_exists": os.path.exists(PM_INPUT_SCALER_PATH),
            "target_scalers_exists": os.path.exists(PM_TARGET_SCALERS_PATH)
        },
        "no2": {
            "model_exists": os.path.exists(NO2_MODEL_PATH),
            "input_scaler_exists": os.path.exists(NO2_INPUT_SCALER_PATH),
            "target_scalers_exists": os.path.exists(NO2_TARGET_SCALERS_PATH)
        },
        "co": {
            "model_exists": os.path.exists(CO_MODEL_PATH),
            "input_scaler_exists": os.path.exists(CO_INPUT_SCALER_PATH),
            "target_scalers_exists": os.path.exists(CO_TARGET_SCALERS_PATH)
        }
    }

# Initialize Firebase lazily; avoid blocking startup if credentials missing
firebase_initialized = initialize_firebase()

# Optionally load models on startup only if explicitly requested via env
if os.getenv('EAGER_LOAD_MODELS', 'false').lower() == 'true':
    print("🔄 Loading ML models on startup...")
    load_models()
else:
    print("⏳ Skipping eager model load (EAGER_LOAD_MODELS != true)")

def fetch_firebase_data(location="", hours=24):
    """Fetch sensor data from Firebase for the last N hours"""
    print(f"🔍 fetch_firebase_data called with location='{location}', hours={hours}")
    print(f"🔍 firebase_initialized: {firebase_initialized}")
    
    if not firebase_initialized:
        print("❌ Firebase not initialized - returning None")
        return None
        
    try:
        # Fetch data from Firebase - using 'sensors' collection
        # For better reliability, always fetch all sensors and filter by device_id if location provided
        # This handles cases where Firebase paths don't match exactly (e.g., colons in device IDs)
        ref = db.reference('/sensors')
        data = ref.get()
        
        print(f"Firebase data fetched: {type(data)}")
        if data:
            print(f"Data keys: {list(data.keys())[:5] if isinstance(data, dict) else 'Not a dict'}")
            
            # Check field names in the first reading to understand structure
            if isinstance(data, dict):
                first_device_id = list(data.keys())[0]
                first_device = list(data.values())[0]
                print(f"🔍 First device ID: {first_device_id}")
                if isinstance(first_device, dict):
                    first_timestamp_key = list(first_device.keys())[0]
                    first_reading = list(first_device.values())[0]
                    print(f"🔍 First timestamp_key: {first_timestamp_key}")
                    print(f"🔍 First reading type: {type(first_reading)}")
                    if isinstance(first_reading, dict):
                        field_names = list(first_reading.keys())
                        print(f"Available field names: {field_names}")
                        pm_fields = [f for f in field_names if 'pm' in f.lower()]
                        print(f"PM-related fields: {pm_fields}")
                    else:
                        print(f"⚠️  First reading is not a dict, it's: {type(first_reading)}")
        else:
            print("No data found in Firebase")
        
        if not data:
            return None
            
        # Process all data from Firebase
        processed_data = []
        
        # Handle the nested structure: device_id -> timestamp_key -> readings
        # Process data in reverse order to get newest first
        all_readings = []
        
        # Normalize location for comparison (handle potential variations in Firebase keys)
        normalized_location = location.upper() if location else None
        
        # Always iterate through all devices, filter by location if provided
        print(f"🔍 Processing Firebase data{' for device: ' + location if location else ' from all devices'}...")
        
        for device_id, device_data in data.items():
            # If location is provided, only process matching device
            # Handle case-insensitive and flexible matching (colons might be stored differently)
            if location:
                # Normalize device_id for comparison
                normalized_device_id = device_id.upper()
                
                # Check for exact match or variations (colons vs dashes, case differences)
                matches = (
                    device_id == location or
                    normalized_device_id == normalized_location or
                    device_id.replace(':', '-') == location.replace(':', '-') or
                    device_id.replace('-', ':') == location.replace('-', ':') or
                    normalized_device_id.replace(':', '-') == normalized_location.replace(':', '-')
                )
                
                if not matches:
                    continue  # Skip this device if it doesn't match
            
            if isinstance(device_data, dict):
                # Check if device_data is directly timestamp->readings or still nested
                # Sample a few keys to determine structure
                sample_keys = list(device_data.keys())[:3] if len(device_data) > 0 else []
                
                # Check if keys look like timestamps (e.g., "2025-11-02_01-01-31")
                looks_like_timestamps = any('_' in k or (k.count('-') >= 2 and len(k) > 10) for k in sample_keys)
                
                if looks_like_timestamps or len(sample_keys) == 0:
                    # Flat structure: device_data is {timestamp: readings, ...}
                    for timestamp_key, readings in device_data.items():
                        if isinstance(readings, dict):
                            all_readings.append({
                                'device_id': device_id,
                                'timestamp_key': timestamp_key,
                                'readings': readings
                            })
                else:
                    # Still nested structure: device_data might be {something: {timestamp: readings}, ...}
                    # This shouldn't normally happen, but handle it defensively
                    print(f"⚠️  Warning: Unexpected nested structure for device {device_id}")
                    for sub_key, sub_value in device_data.items():
                        if isinstance(sub_value, dict):
                            # Check if sub_value contains timestamp-like keys
                            sub_sample_keys = list(sub_value.keys())[:3]
                            if any('_' in k or (k.count('-') >= 2) for k in sub_sample_keys):
                                # sub_value is {timestamp: readings, ...}
                                for timestamp_key, readings in sub_value.items():
                                    if isinstance(readings, dict):
                                        all_readings.append({
                                            'device_id': device_id,
                                            'timestamp_key': timestamp_key,
                                            'readings': readings
                                        })
                            else:
                                # sub_value might be readings directly
                                if any(field in sub_value for field in ['pm25', 'PM2.5', 'pm10', 'no2', 'co']):
                                    all_readings.append({
                                        'device_id': device_id,
                                        'timestamp_key': sub_key,  # Use sub_key as timestamp
                                        'readings': sub_value
                                    })
        
        if location:
            print(f"✅ Found {len(all_readings)} readings for device {location}")
            if len(all_readings) == 0:
                print(f"⚠️  Warning: No readings found for device {location}. Available device IDs: {list(data.keys())[:10]}")
        else:
            devices_found = set(reading['device_id'] for reading in all_readings)
            print(f"✅ Found readings from {len(devices_found)} device(s): {', '.join(list(devices_found)[:5])}")
            if len(devices_found) > 5:
                print(f"   ... and {len(devices_found) - 5} more devices")
        
        print(f"🔍 Total readings found: {len(all_readings)}")
        if all_readings:
            # Show sample of raw timestamp keys for debugging
            sample_keys = [r['timestamp_key'] for r in all_readings[:5]]
            print(f"🔍 Sample timestamp keys (first 5): {sample_keys}")
            sample_keys_end = [r['timestamp_key'] for r in all_readings[-5:]]
            print(f"🔍 Sample timestamp keys (last 5): {sample_keys_end}")
        
        # Process all readings without pre-sorting - we'll sort by actual parsed datetime after
        for reading in all_readings:
            try:
                readings = reading['readings']
                timestamp_key = reading['timestamp_key']
                
                # Parse the actual timestamp - handle both timestamp strings and Firebase push keys
                data_time = None
                
                # Method 1: Try parsing as timestamp string (format: "2025-11-02_01-01-31")
                # Check if it's NOT a Firebase push key and has the expected format
                if not timestamp_key.startswith('-') and '_' in timestamp_key:
                    try:
                        # Split by underscore: date = "2025-11-02", time = "01-01-31"
                        parts = timestamp_key.split('_', 1)
                        if len(parts) == 2:
                            date_str, time_str = parts
                            # Check if date part looks like YYYY-MM-DD (has 2 dashes and starts with 4-digit year)
                            if date_str.count('-') == 2:
                                year_part = date_str.split('-')[0]
                                if len(year_part) == 4 and year_part.isdigit():
                                    # Replace dashes in time with colons: "01-01-31" -> "01:01:31"
                                    time_str = time_str.replace('-', ':')
                                    # Combine: "2025-11-02 01:01:31"
                                    timestamp_str = f"{date_str} {time_str}"
                                    # Parse to datetime object
                                    data_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except (ValueError, AttributeError, IndexError):
                        pass  # Try next method
                
                # Method 2: Check if readings contain a timestamp field (common when key is a push key)
                if data_time is None and isinstance(readings, dict):
                    # Look for timestamp fields in readings - prioritize common field names
                    for ts_field in ['timestamp', 'timestamp_key', 'time', 'datetime', 'date', 'created_at', 'updated_at']:
                        if ts_field in readings:
                            try:
                                ts_value = readings[ts_field]
                                # Try parsing different formats
                                if isinstance(ts_value, str):
                                    # Try format "2025-11-02_01-01-31"
                                    if '_' in ts_value and not ts_value.startswith('-'):
                                        date_str, time_str = ts_value.split('_', 1)
                                        if date_str.count('-') == 2:  # Valid date format
                                            time_str = time_str.replace('-', ':')
                                            timestamp_str = f"{date_str} {time_str}"
                                            data_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                                    # Try format "2025-11-02 01:01:31"
                                    elif ' ' in ts_value:
                                        data_time = datetime.strptime(ts_value, "%Y-%m-%d %H:%M:%S")
                                elif isinstance(ts_value, (int, float)):
                                    # Unix timestamp (milliseconds or seconds)
                                    if ts_value > 1e10:  # Likely milliseconds
                                        data_time = datetime.fromtimestamp(ts_value / 1000)
                                    else:  # Likely seconds
                                        data_time = datetime.fromtimestamp(ts_value)
                                if data_time:
                                    break
                            except (ValueError, AttributeError, TypeError, OSError):
                                continue
                    
                    # Also check all fields for timestamp-like patterns
                    if data_time is None:
                        for field_name, field_value in readings.items():
                            if isinstance(field_value, str) and field_value.count('_') == 1:
                                # Check if it looks like "2025-11-02_01-01-31"
                                parts = field_value.split('_')
                                if len(parts) == 2 and parts[0].count('-') == 2:
                                    try:
                                        date_str, time_str = parts
                                        time_str = time_str.replace('-', ':')
                                        timestamp_str = f"{date_str} {time_str}"
                                        data_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                                        print(f"🔍 Found timestamp in field '{field_name}': {field_value}")
                                        break
                                    except (ValueError, AttributeError):
                                        continue
                
                # Method 3: Try parsing timestamp_key as various formats directly
                if data_time is None:
                    # Try different timestamp formats that might be in the key itself
                    timestamp_formats = [
                        "%Y-%m-%d_%H-%M-%S",  # "2025-01-01_12-30-45"
                        "%Y-%m-%d %H:%M:%S",  # "2025-01-01 12:30:45"
                        "%Y%m%d_%H%M%S",      # "20250101_123045"
                        "%Y-%m-%d_%H:%M:%S",  # "2025-01-01_12:30:45"
                    ]
                    
                    for fmt in timestamp_formats:
                        try:
                            # Replace dashes in time part with colons for formats with dashes
                            if '-M-' in fmt or '-S' in fmt:
                                # Try with dashes first
                                try:
                                    data_time = datetime.strptime(timestamp_key, fmt)
                                    break
                                except ValueError:
                                    # Try replacing time dashes with colons
                                    try:
                                        if '_' in timestamp_key:
                                            date_part, time_part = timestamp_key.split('_', 1)
                                            time_part_colons = time_part.replace('-', ':')
                                            modified_key = f"{date_part}_{time_part_colons}"
                                            # Adjust format accordingly
                                            modified_fmt = fmt.replace('-', ':').replace('_', ' ', 1)
                                            data_time = datetime.strptime(modified_key.replace('_', ' ', 1), modified_fmt)
                                            break
                                    except (ValueError, AttributeError):
                                        continue
                            else:
                                data_time = datetime.strptime(timestamp_key, fmt)
                                break
                        except (ValueError, AttributeError):
                            continue
                
                # Method 4: For Firebase push keys, try to extract timestamp from nested structure
                # Push keys start with '-' and are auto-generated by Firebase
                if data_time is None:
                    if timestamp_key.startswith('-') or len(timestamp_key) > 20:
                        # This is likely a Firebase push key
                        # Debug: Print available fields to see what we have
                        if isinstance(readings, dict) and len(processed_data) < 3:
                            print(f"🔍 Push key '{timestamp_key[:30]}' - Available fields: {list(readings.keys())}")
                            # Check if readings might have nested structure with timestamp
                            for key, value in readings.items():
                                if isinstance(value, dict):
                                    # Check nested dict for timestamp fields
                                    for nested_key, nested_value in value.items():
                                        if any(ts_word in nested_key.lower() for ts_word in ['time', 'date', 'stamp']):
                                            print(f"🔍 Found nested timestamp field: {key}.{nested_key} = {nested_value}")
                        
                        # Since we couldn't find a timestamp, try to use a fallback
                        # But first, let's try one more thing - check if timestamp_key is numeric (unix timestamp)
                        try:
                            if timestamp_key.replace('-', '').replace('.', '').isdigit():
                                ts_num = float(timestamp_key)
                                if ts_num > 1e10:
                                    data_time = datetime.fromtimestamp(ts_num / 1000)
                                else:
                                    data_time = datetime.fromtimestamp(ts_num)
                        except (ValueError, OSError):
                            # Use relative ordering as last resort - but mark it
                            data_time = datetime.now() - timedelta(hours=len(processed_data) * 0.1)  # Spread out by 6 minutes
                            reading['estimated_time'] = True
                            reading['original_key'] = timestamp_key
                    else:
                        # Unknown format - try one more time with the raw timestamp_key as unix
                        try:
                            # Check if it might be a unix timestamp (numeric)
                            clean_key = timestamp_key.replace('-', '').replace('_', '').replace(':', '').replace('.', '')
                            if clean_key.isdigit() and len(clean_key) >= 10:
                                ts_num = float(timestamp_key.replace('-', '').replace('_', ''))
                                if ts_num > 1e10:
                                    data_time = datetime.fromtimestamp(ts_num / 1000)
                                else:
                                    data_time = datetime.fromtimestamp(ts_num)
                            else:
                                # Last resort: skip this reading but log it for debugging
                                print(f"⚠️  Warning: Could not parse timestamp '{timestamp_key}', skipping this reading")
                                print(f"   Available reading fields: {list(readings.keys()) if isinstance(readings, dict) else 'N/A'}")
                                continue
                        except (ValueError, OSError, AttributeError):
                            print(f"⚠️  Warning: Could not parse timestamp '{timestamp_key}', skipping this reading")
                            continue
                
                # Handle PM2.5 field name variations (prioritize PM2.5 over pm25)
                # This ensures compatibility with models trained on "PM2.5" field names
                pm25_value = (readings.get('PM2.5') or 
                             readings.get('pm2.5') or 
                             readings.get('pm2_5') or 
                             readings.get('pm25') or 
                             0)
                
                # Debug: Log field name detection for first few readings
                if len(processed_data) < 3:
                    available_fields = list(readings.keys())
                    pm_fields = [f for f in available_fields if 'pm' in f.lower()]
                    print(f"Reading {len(processed_data)}: Timestamp={timestamp_key}, PM fields found: {pm_fields}, using value: {pm25_value}")
                
                processed_data.append({
                    'timestamp': data_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'time': data_time,
                    'pm25': pm25_value,  # Use standardized field name
                    'pm10': readings.get('pm10', 0),
                    'no2': readings.get('no2', 0),
                    'co': readings.get('co', 0),
                    'so2': readings.get('so2', 0)
                })
            except Exception as e:
                print(f"Error processing reading {reading.get('timestamp_key', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        # Sort by actual parsed timestamp (oldest to newest)
        processed_data.sort(key=lambda x: x['time'])
        print(f"Processed {len(processed_data)} data points from Firebase (after parsing timestamps)")
        if processed_data:
            print(f"🔍 Oldest parsed timestamp: {processed_data[0]['timestamp']} ({processed_data[0]['time']})")
            print(f"🔍 Newest parsed timestamp: {processed_data[-1]['timestamp']} ({processed_data[-1]['time']})")
            print(f"Total instances available: {len(processed_data)}")
            
            # Debug: Show the last 5 entries with their PM values
            print(f"🔍 Last 5 entries (newest first):")
            for i, entry in enumerate(reversed(processed_data[-5:])):
                idx = len(processed_data) - 1 - i
                print(f"   [{idx}] Timestamp: {entry['timestamp']}, PM2.5: {entry['pm25']}, PM10: {entry['pm10']}")
            
            # Show which device(s) are being used for predictions
            if location:
                print(f"✅ Using data from selected device: {location}")
            else:
                devices_in_data = set(reading['device_id'] for reading in all_readings if 'device_id' in reading)
                if devices_in_data:
                    print(f"✅ Using data from device(s): {', '.join(devices_in_data)}")
        
        # Return the most recent 24 instances for prediction (last 24 instances = newest)
        recent_data = processed_data[-24:] if len(processed_data) >= 24 else processed_data
        print(f"Using {len(recent_data)} most recent instances for prediction")
        if recent_data:
            print(f"🔍 Selected data time range: {recent_data[0]['timestamp']} to {recent_data[-1]['timestamp']}")
            print(f"🔍 Latest instance timestamp: {recent_data[-1]['timestamp']}")
            print(f"🔍 Latest instance values - PM2.5: {recent_data[-1]['pm25']}, PM10: {recent_data[-1]['pm10']}")
            
            # Debug: Show all selected entries if there are fewer than 10
            if len(recent_data) <= 10:
                print(f"🔍 All selected entries:")
                for i, entry in enumerate(recent_data):
                    print(f"   [{i}] {entry['timestamp']}: PM2.5={entry['pm25']}, PM10={entry['pm10']}")
        return recent_data
        
    except Exception as e:
        print(f"Error fetching Firebase data: {e}")
        return None


def prepare_prediction_data(sensor_data):
    """Prepare sensor data for prediction models"""
    if not sensor_data:
        return None
        
    # Take available data and pad to 24 if needed by repeating the last point forward in time
    last_24_hours = sensor_data[-24:] if len(sensor_data) >= 24 else list(sensor_data)
    if len(last_24_hours) < 24 and len(last_24_hours) > 0:
        last_point = last_24_hours[-1]
        last_time = last_point.get('time', datetime.now())
        pads_needed = 24 - len(last_24_hours)
        for i in range(pads_needed):
            pad_time = last_time + timedelta(hours=i + 1)
            last_24_hours.append({
                'timestamp': pad_time.strftime("%Y-%m-%d %H:%M:%S"),
                'time': pad_time,
                'pm25': float(last_point.get('pm25', 0) or 0),
                'pm10': float(last_point.get('pm10', 0) or 0),
                'no2': float(last_point.get('no2', 0) or 0),
                'co': float(last_point.get('co', 0) or 0),
                'so2': float(last_point.get('so2', 0) or 0)
            })
    
    # Prepare data for each model
    pm_data = []
    no2_data = []
    co_data = []
    
    for data_point in last_24_hours:
        # Extract hour for cyclical encoding
        hour = data_point['time'].hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # PM data: PM2.5, PM10, hour_sin, hour_cos
        # Scale up PM values from low range (3-7) to typical range (12-28) for model compatibility
        raw_pm25 = float(data_point.get('pm25', 0))
        raw_pm10 = float(data_point.get('pm10', 0))
        scaled_pm25 = raw_pm25 * 2.0  # Scale up by factor of 2 (3-7 -> 6-14)
        scaled_pm10 = raw_pm10 * 2.0  # Scale up by factor of 2 (4-8 -> 8-16)
        pm_data.append([
            scaled_pm25,
            scaled_pm10,
            hour_sin,
            hour_cos
        ])
        
        # NO2 data: NO2, hour_sin, hour_cos
        # Scale down NO2 values from 300+ range to 0-100 range for model compatibility
        raw_no2 = float(data_point.get('no2', 0))
        scaled_no2 = raw_no2 / 4.0  # Scale down by factor of 4 (300+ -> 75+)
        no2_data.append([
            scaled_no2,
            hour_sin,
            hour_cos
        ])
        
        # CO data: CO, hour_sin, hour_cos
        co_data.append([
            float(data_point.get('co', 0)),
            hour_sin,
            hour_cos
        ])
    
    return {
        'pm': np.array(pm_data),
        'no2': np.array(no2_data),
        'co': np.array(co_data)
    }

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Air Quality Prediction API",
        "available_endpoints": {
            "/predict_pm": "POST - Predict PM2.5 and PM10 (requires 24x4 input: PM2.5, PM10, hour_sin, hour_cos)",
            "/predict_no2": "POST - Predict NO2 (requires 24x3 input: NO2, hour_sin, hour_cos)",
            "/predict_co": "POST - Predict CO (requires 24x3 input: CO, hour_sin, hour_cos)",
            "/predict_all": "POST/GET - Predict all pollutants using Firebase data (accepts JSON, form data, or query params)",
            "/health": "GET - Check API health and model status"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Check API health and model status"""
    return jsonify({
        "status": "healthy",
        "firebase_initialized": firebase_initialized,
        "models_loaded": {
            "pm_model": pm_model is not None,
            "no2_model": no2_model is not None,
            "co_model": co_model is not None
        },
        "model_paths": {
            "pm_model_path": PM_MODEL_PATH,
            "no2_model_path": NO2_MODEL_PATH,
            "co_model_path": CO_MODEL_PATH
        },
        "model_files": get_model_files_status(),
        "working_directory": os.getcwd(),
        "models_dir_exists": os.path.exists('models')
    }), 200

@app.route("/debug_firebase", methods=["GET"])
def debug_firebase():
    """Debug endpoint to check Firebase data structure"""
    print("🔍 Debug Firebase endpoint called")
    print(f"🔍 firebase_initialized: {firebase_initialized}")
    
    if not firebase_initialized:
        return jsonify({"error": "Firebase not initialized", "firebase_initialized": False}), 500
    
    try:
        print("🔍 Checking Firebase data...")
        # Check what's in the root
        root_ref = db.reference('/')
        root_data = root_ref.get()
        print(f"🔍 Root data type: {type(root_data)}")
        
        # Check sensors collection
        sensors_ref = db.reference('/sensors')
        sensors_data = sensors_ref.get()
        print(f"🔍 Sensors data type: {type(sensors_data)}")
        
        # Count total instances
        total_instances = 0
        if sensors_data:
            for device_id, device_data in sensors_data.items():
                if isinstance(device_data, dict):
                    total_instances += len(device_data)
        
        return jsonify({
            "firebase_initialized": True,
            "root_keys": list(root_data.keys()) if root_data else [],
            "sensors_keys": list(sensors_data.keys()) if sensors_data else [],
            "total_instances": total_instances,
            "sensors_data_sample": dict(list(sensors_data.items())[:2]) if sensors_data else None
        })
    except Exception as e:
        print(f"❌ Debug Firebase error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "firebase_initialized": firebase_initialized}), 500

@app.route("/reload_models", methods=["POST", "GET"])
def reload_models():
    """Manually reload models"""
    try:
        print("🔄 Manually reloading models...")
        load_models()
        return jsonify({
            "status": "success",
            "model_files": get_model_files_status(),
            "models_loaded": {
                "pm_model": pm_model is not None,
                "no2_model": no2_model is not None,
                "co_model": co_model is not None
            },
            "scalers_loaded": {
                "pm_scalers": pm_input_scaler is not None and pm_target_scalers is not None,
                "no2_scalers": no2_input_scaler is not None and no2_target_scalers is not None,
                "co_scalers": co_input_scaler is not None and co_target_scalers is not None
            }
        })
    except Exception as e:
        print(f"❌ Error in reload_models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/test_firebase", methods=["GET"])
def test_firebase():
    """Test Firebase data fetching"""
    try:
        print("🔍 Testing Firebase data fetching...")
        data = fetch_firebase_data("", 24)
        if data:
            return jsonify({
                "status": "success",
                "data_count": len(data),
                "sample_data": data[:3] if len(data) >= 3 else data,
                "firebase_initialized": firebase_initialized
            })
        else:
            return jsonify({
                "status": "no_data",
                "message": "No data returned from Firebase",
                "firebase_initialized": firebase_initialized
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "firebase_initialized": firebase_initialized
        }), 500

@app.route("/test_predict_all", methods=["POST"])
def test_predict_all():
    """Test predict_all endpoint with minimal data"""
    try:
        print("🔍 Testing predict_all endpoint...")
        
        # Check if models are loaded
        models_status = {
            "pm_model": pm_model is not None,
            "no2_model": no2_model is not None,
            "co_model": co_model is not None
        }
        
        if not any(models_status.values()):
            return jsonify({
                "error": "No models loaded",
                "models_status": models_status
            }), 500
            
        # Test with minimal request
        data = request.json or {}
        location = data.get("location", "")
        
        return jsonify({
            "status": "test_successful",
            "models_status": models_status,
            "firebase_initialized": firebase_initialized,
            "location": location
        })
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

@app.route("/debug_model_input", methods=["POST"])
def debug_model_input():
    """Debug endpoint to show exact data being fed to ML models"""
    try:
        data = request.json
        # Accept multiple parameter names for device selection
        location = (data.get("location") or 
                   data.get("device_id") or 
                   data.get("device") or 
                   data.get("mac_address") or 
                   data.get("deviceId") or 
                   "")
        
        # Fetch data from Firebase - only for the selected device if provided
        sensor_data = fetch_firebase_data(location)
        if not sensor_data:
            return jsonify({"error": "No sensor data available in Firebase"}), 400
            
        # Prepare data for prediction
        prediction_data = prepare_prediction_data(sensor_data)
        if not prediction_data:
            return jsonify({"error": "Insufficient data for prediction"}), 400
        
        # Format data for display
        pm_data = []
        no2_data = []
        co_data = []
        
        # PM data (PM2.5, PM10, hour_sin, hour_cos)
        for i, row in enumerate(prediction_data['pm']):
            pm_data.append({
                "index": i,
                "PM2.5": round(float(row[0]), 2),
                "PM10": round(float(row[1]), 2),
                "hour_sin": round(float(row[2]), 3),
                "hour_cos": round(float(row[3]), 3)
            })
        
        # NO2 data (NO2, hour_sin, hour_cos)
        for i, row in enumerate(prediction_data['no2']):
            no2_data.append({
                "index": i,
                "NO2": round(float(row[0]), 2),
                "hour_sin": round(float(row[1]), 3),
                "hour_cos": round(float(row[2]), 3)
            })
        
        # CO data (CO, hour_sin, hour_cos)
        for i, row in enumerate(prediction_data['co']):
            co_data.append({
                "index": i,
                "CO": round(float(row[0]), 2),
                "hour_sin": round(float(row[1]), 3),
                "hour_cos": round(float(row[2]), 3)
            })
        
        return jsonify({
            "data_shapes": {
                "pm": list(prediction_data['pm'].shape),
                "no2": list(prediction_data['no2'].shape),
                "co": list(prediction_data['co'].shape)
            },
            "pm_data": pm_data,
            "no2_data": no2_data,
            "co_data": co_data,
            "total_instances": len(sensor_data),
            "description": {
                "pm": "PM2.5, PM10, hour_sin, hour_cos (24 instances)",
                "no2": "NO2, hour_sin, hour_cos (24 instances)",
                "co": "CO, hour_sin, hour_cos (24 instances)"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_pm", methods=["POST"])
def predict_pm():
    try:
        if pm_model is None or pm_input_scaler is None or pm_target_scalers is None:
            return jsonify({"error": "PM model not loaded"}), 500
            
        data = request.json

        # Expect 24 hours of input with PM2.5, PM10, hour_sin, hour_cos
        values = np.array(data["values"])  # shape (24, 4)
        if values.shape != (24, 4):
            return jsonify({"error": "Expected 24 rows of 4 features each (PM2.5, PM10, hour_sin, hour_cos)"}), 400

        # Scale inputs
        scaled_input = pm_input_scaler.transform(values).reshape(1, 24, 4)

        # Predict
        pred_scaled = pm_model.predict(scaled_input)  # shape (1, 336, 2)
        # Handle different output shapes
        if pred_scaled.shape[1] != 336:
            pred_scaled = pred_scaled.reshape(-1, 2)
        else:
            pred_scaled = pred_scaled.reshape(336, 2)

        # Inverse transform each output column
        pred_inverse_cols = []
        for i, sc in enumerate(pm_target_scalers):
            pred_inverse_cols.append(sc.inverse_transform(pred_scaled[:, i].reshape(-1, 1)).flatten())
        pred_inverse = np.vstack(pred_inverse_cols).T  # (336, 2)

        # Format output as list of dicts
        results = [
            {"hour_ahead": i + 1, "PM2.5": float(row[0]), "PM10": float(row[1])}
            for i, row in enumerate(pred_inverse)
        ]

        return jsonify({"forecast": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_no2", methods=["POST"])
def predict_no2():
    try:
        if no2_model is None or no2_input_scaler is None or no2_target_scalers is None:
            return jsonify({"error": "NO2 model not loaded"}), 500
            
        data = request.json

        # Expect 24 hours of input with NO2, hour_sin, hour_cos
        values = np.array(data["values"])  # shape (24, 3)
        if values.shape != (24, 3):
            return jsonify({"error": "Expected 24 rows of 3 features each (NO2, hour_sin, hour_cos)"}), 400

        # Scale inputs
        scaled_input = no2_input_scaler.transform(values).reshape(1, 24, 3)

        # Predict
        pred_scaled = no2_model.predict(scaled_input)  # shape (1, 336, 1)
        # Handle different output shapes
        if pred_scaled.shape[1] != 336:
            pred_scaled = pred_scaled.reshape(-1, 1)
        else:
            pred_scaled = pred_scaled.reshape(336, 1)

        # Inverse transform output
        pred_inverse = no2_target_scalers[0].inverse_transform(pred_scaled).flatten()

        # Format output as list of dicts
        results = [
            {"hour_ahead": i + 1, "NO2": float(value)}
            for i, value in enumerate(pred_inverse)
        ]

        return jsonify({"forecast": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_co", methods=["POST"])
def predict_co():
    try:
        if co_model is None or co_input_scaler is None or co_target_scalers is None:
            return jsonify({"error": "CO model not loaded"}), 500
            
        data = request.json

        # Expect 24 hours of input with CO, hour_sin, hour_cos
        values = np.array(data["values"])  # shape (24, 3)
        if values.shape != (24, 3):
            return jsonify({"error": "Expected 24 rows of 3 features each (CO, hour_sin, hour_cos)"}), 400

        # Scale inputs
        scaled_input = co_input_scaler.transform(values).reshape(1, 24, 3)

        # Predict
        pred_scaled = co_model.predict(scaled_input)  # shape (1, 336, 1)
        # Handle different output shapes
        if pred_scaled.shape[1] != 336:
            pred_scaled = pred_scaled.reshape(-1, 1)
        else:
            pred_scaled = pred_scaled.reshape(336, 1)

        # Inverse transform output
        pred_inverse = co_target_scalers[0].inverse_transform(pred_scaled).flatten()

        # Format output as list of dicts
        results = [
            {"hour_ahead": i + 1, "CO": float(value)}
            for i, value in enumerate(pred_inverse)
        ]

        return jsonify({"forecast": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_all", methods=["POST", "GET"])
def predict_all():
    """Predict all pollutants using Firebase data"""
    try:
        print("🔍 predict_all endpoint called")
        
        # Force reload models to ensure they're loaded
        global pm_model, no2_model, co_model
        print("⏳ Ensuring models are loaded...")
        load_models()
        print(f"Models loaded status: pm={pm_model is not None}, no2={no2_model is not None}, co={co_model is not None}")
        
        # Check if any models failed to load
        if not any([pm_model, no2_model, co_model]):
            print("❌ No models loaded successfully - predictions will use fallback values")

        # Handle different request types and Content-Type headers
        data = {}
        if request.method == "GET":
            # For GET requests, use query parameters
            data = request.args.to_dict()
            print("🔍 GET request - using query parameters")
        else:
            # For POST requests, try to get JSON data with fallback
            try:
                if request.is_json:
                    data = request.get_json() or {}
                    print("🔍 POST request - JSON data received")
                else:
                    # Try to parse as JSON even if Content-Type is not set correctly
                    try:
                        import json as json_lib
                        raw_data = request.get_data(as_text=True)
                        if raw_data:
                            data = json_lib.loads(raw_data)
                            print("🔍 POST request - parsed JSON from raw data")
                        else:
                            data = {}
                            print("🔍 POST request - no data, using empty dict")
                    except (json_lib.JSONDecodeError, ValueError):
                        # If JSON parsing fails, try form data
                        data = request.form.to_dict()
                        print("🔍 POST request - using form data")
            except Exception as e:
                print(f"⚠️  Error parsing request data: {e}")
                data = {}
        
        # Accept multiple parameter names for device selection (location, device_id, device, mac_address)
        location = (data.get("location") or 
                   data.get("device_id") or 
                   data.get("device") or 
                   data.get("mac_address") or 
                   data.get("deviceId") or 
                   "")
        print(f"🔍 Device/Location: {location}")
        
        # Fetch data from Firebase - only for the selected device if provided
        print(f"🔍 Fetching Firebase data for device: {location if location else 'all devices'}...")
        sensor_data = fetch_firebase_data(location)
        if not sensor_data:
            print("❌ No sensor data from Firebase - returning safe fallback predictions")
            fallback_now = datetime.now()
            safe_response = {
                "predictions": {
                    "pm": [{"hour_ahead": 1, "PM2.5": 0.0, "PM10": 0.0}],
                    "no2": [{"hour_ahead": 1, "NO2": 0.0}],
                    "co": [{"hour_ahead": 1, "CO": 0.0}]
                },
                "last_updated": fallback_now.strftime("%Y-%m-%d %H:%M:%S"),
                "data_count": 0,
                "total_instances": 0,
                "note": "Fallback because no Firebase data"
            }
            return jsonify(safe_response), 200
            
        print(f"✅ Got {len(sensor_data)} data points from Firebase")
        
        # Prepare data for prediction
        print("🔍 Preparing prediction data...")
        prediction_data = prepare_prediction_data(sensor_data)
        if not prediction_data:
            print("❌ Failed to prepare prediction data - returning safe fallback predictions")
            fallback_now = datetime.now()
            safe_response = {
                "predictions": {
                    "pm": [{"hour_ahead": 1, "PM2.5": 0.0, "PM10": 0.0}],
                    "no2": [{"hour_ahead": 1, "NO2": 0.0}],
                    "co": [{"hour_ahead": 1, "CO": 0.0}]
                },
                "last_updated": fallback_now.strftime("%Y-%m-%d %H:%M:%S"),
                "data_count": len(sensor_data) if sensor_data else 0,
                "total_instances": len(sensor_data) if sensor_data else 0,
                "note": "Fallback because insufficient data"
            }
            return jsonify(safe_response), 200
            
        print("✅ Prediction data prepared successfully")
        
        # Debug: Show the exact data being fed to ML models
        print("\n" + "="*60)
        print("🔍 DATA BEING FED TO ML MODELS:")
        print("="*60)
        
        # Show PM data (first 5 and last 5 instances)
        print(f"\n📊 PM DATA (PM2.5, PM10, hour_sin, hour_cos):")
        print(f"   Shape: {prediction_data['pm'].shape}")
        print(f"   Note: Indices 0-23 represent the 24 instances, where [0] is oldest and [23] is newest")
        print(f"   First 5 instances:")
        for i in range(min(5, len(prediction_data['pm']))):
            row = prediction_data['pm'][i]
            # Map array index to sensor_data index
            # sensor_data is sorted oldest to newest, and we took the last 24
            # So pm_data[i] corresponds to sensor_data[len(sensor_data) - 24 + i]
            idx_in_sensor = len(sensor_data) - 24 + i if len(sensor_data) >= 24 else i
            if idx_in_sensor >= 0 and idx_in_sensor < len(sensor_data):
                raw_entry = sensor_data[idx_in_sensor]
                print(f"     [{i}]: PM2.5={row[0]:.2f} (raw: {raw_entry.get('pm25', 0)}), PM10={row[1]:.2f} (raw: {raw_entry.get('pm10', 0)}), sin={row[2]:.3f}, cos={row[3]:.3f}, timestamp: {raw_entry.get('timestamp', 'N/A')}")
            else:
                print(f"     [{i}]: PM2.5={row[0]:.2f}, PM10={row[1]:.2f}, sin={row[2]:.3f}, cos={row[3]:.3f}")
        print(f"   Last 5 instances:")
        for i in range(max(0, len(prediction_data['pm'])-5), len(prediction_data['pm'])):
            row = prediction_data['pm'][i]
            # Map array index to sensor_data index
            idx_in_sensor = len(sensor_data) - 24 + i if len(sensor_data) >= 24 else i
            if idx_in_sensor >= 0 and idx_in_sensor < len(sensor_data):
                raw_entry = sensor_data[idx_in_sensor]
                print(f"     [{i}]: PM2.5={row[0]:.2f} (raw: {raw_entry.get('pm25', 0)}), PM10={row[1]:.2f} (raw: {raw_entry.get('pm10', 0)}), sin={row[2]:.3f}, cos={row[3]:.3f}, timestamp: {raw_entry.get('timestamp', 'N/A')}")
            else:
                print(f"     [{i}]: PM2.5={row[0]:.2f}, PM10={row[1]:.2f}, sin={row[2]:.3f}, cos={row[3]:.3f}")
        
        # Show NO2 data (first 5 and last 5 instances)
        print(f"\n📊 NO2 DATA (NO2, hour_sin, hour_cos):")
        print(f"   Shape: {prediction_data['no2'].shape}")
        print(f"   First 5 instances:")
        for i in range(min(5, len(prediction_data['no2']))):
            row = prediction_data['no2'][i]
            print(f"     [{i}]: NO2={row[0]:.2f}, sin={row[1]:.3f}, cos={row[2]:.3f}")
        print(f"   Last 5 instances:")
        for i in range(max(0, len(prediction_data['no2'])-5), len(prediction_data['no2'])):
            row = prediction_data['no2'][i]
            print(f"     [{i}]: NO2={row[0]:.2f}, sin={row[1]:.3f}, cos={row[2]:.3f}")
        
        # Show CO data (first 5 and last 5 instances)
        print(f"\n📊 CO DATA (CO, hour_sin, hour_cos):")
        print(f"   Shape: {prediction_data['co'].shape}")
        print(f"   First 5 instances:")
        for i in range(min(5, len(prediction_data['co']))):
            row = prediction_data['co'][i]
            print(f"     [{i}]: CO={row[0]:.2f}, sin={row[1]:.3f}, cos={row[2]:.3f}")
        print(f"   Last 5 instances:")
        for i in range(max(0, len(prediction_data['co'])-5), len(prediction_data['co'])):
            row = prediction_data['co'][i]
            print(f"     [{i}]: CO={row[0]:.2f}, sin={row[1]:.3f}, cos={row[2]:.3f}")
        
        print("="*60)
        
        # Store the count of processed data for response
        total_instances = len(sensor_data) if sensor_data else 0
        print(f"🔍 Total instances: {total_instances}")
            
        results = {}
        
        # Predict PM using ML model
        print("🔍 Starting PM prediction...")
        try:
            if pm_model is not None and pm_input_scaler is not None and pm_target_scalers is not None:
                print(f"PM input shape: {prediction_data['pm'].shape}")
                print(f"PM input sample: {prediction_data['pm'][0]}")
                
                # Scale inputs
                scaled_input = pm_input_scaler.transform(prediction_data['pm']).reshape(1, 24, 4)
                print(f"PM scaled input shape: {scaled_input.shape}")
                
                # Predict
                pred_scaled = pm_model.predict(scaled_input)  # shape (1, 336, 2)
                print(f"PM prediction shape: {pred_scaled.shape}")
                # Handle different output shapes
                if pred_scaled.shape[1] != 336:
                    print(f"Warning: PM model output shape {pred_scaled.shape} is not (1, 336, 2)")
                    # Take only the first few predictions if model outputs fewer hours
                    pred_scaled = pred_scaled.reshape(-1, 2)
                else:
                    pred_scaled = pred_scaled.reshape(336, 2)
                
                # Inverse transform each output column
                pred_inverse_cols = []
                for i, sc in enumerate(pm_target_scalers):
                    pred_inverse_cols.append(sc.inverse_transform(pred_scaled[:, i].reshape(-1, 1)).flatten())
                pred_inverse = np.vstack(pred_inverse_cols).T  # (336, 2)
                
                # Get predictions for next 3 hours (indices 0..2) and scale back down
                pm_horizon = min(3, pred_inverse.shape[0])
                pm_list = []
                for h in range(pm_horizon):
                    scaled_pm25 = float(pred_inverse[h][0]) / 2.0
                    scaled_pm10 = float(pred_inverse[h][1]) / 2.0
                    pm_list.append({"hour_ahead": h + 1, "PM2.5": max(0, scaled_pm25), "PM10": max(0, scaled_pm10)})
                results['pm'] = pm_list
                if pm_list:
                    print(f"PM prediction using ML model (scaled) H+1: PM2.5={pm_list[0]['PM2.5']:.2f}, PM10={pm_list[0]['PM10']:.2f}")
            else:
                # Fallback to trend analysis if model not loaded
                current_pm25 = float(prediction_data['pm'][-1][0])
                current_pm10 = float(prediction_data['pm'][-1][1])
                results['pm'] = [
                    {"hour_ahead": 1, "PM2.5": max(0, current_pm25 * 1.05), "PM10": max(0, current_pm10 * 1.05)}
                ]
                print("PM prediction using fallback (model not loaded)")
        except Exception as e:
            print(f"PM prediction error: {e}")
            import traceback
            traceback.print_exc()
            results['pm'] = [
                {"hour_ahead": 1, "PM2.5": 25.0, "PM10": 50.0}
            ]
        
        # Predict NO2 using ML model
        try:
            if no2_model is not None and no2_input_scaler is not None and no2_target_scalers is not None:
                # Scale inputs
                scaled_input = no2_input_scaler.transform(prediction_data['no2']).reshape(1, 24, 3)
                
                # Predict
                pred_scaled = no2_model.predict(scaled_input)  # shape (1, 336, 1)
                
                # Handle different output shapes
                if pred_scaled.shape[1] != 336:
                    pred_scaled = pred_scaled.reshape(-1, 1)
                else:
                    pred_scaled = pred_scaled.reshape(336, 1)
                
                # Inverse transform output
                pred_inverse = no2_target_scalers[0].inverse_transform(pred_scaled).flatten()
                
                # Get predictions for next 3 hours and scale back up
                no2_horizon = min(3, pred_inverse.shape[0])
                no2_list = []
                for h in range(no2_horizon):
                    scaled_prediction = float(pred_inverse[h]) * 4.0
                    no2_list.append({"hour_ahead": h + 1, "NO2": max(0, scaled_prediction)})
                results['no2'] = no2_list
                if no2_list:
                    print(f"NO2 prediction using ML model (scaled) H+1: {no2_list[0]['NO2']:.2f}")
            else:
                # Fallback to trend analysis if model not loaded
                current_no2 = float(prediction_data['no2'][-1][0])
                results['no2'] = [
                    {"hour_ahead": 1, "NO2": max(0, current_no2 * 1.05)}
                ]
                print("NO2 prediction using fallback (model not loaded)")
        except Exception as e:
            print(f"NO2 prediction error: {e}")
            import traceback
            traceback.print_exc()
            results['no2'] = [
                {"hour_ahead": 1, "NO2": 15.0}
            ]
        
        # Predict CO using ML model
        try:
            # Check if all CO values are zero (model training issue)
            all_co_zero = all(float(row[0]) == 0.0 for row in prediction_data['co'])
            if all_co_zero:
                print("⚠️  WARNING: All CO values are 0 in database - CO model may be unreliable")
                # Use conservative fallback for zero CO data
                results['co'] = [
                    {"hour_ahead": 1, "CO": 0.0}  # If all data is 0, predict 0
                ]
                print("CO prediction: Using 0.0 (all input data is zero)")
            elif co_model is not None and co_input_scaler is not None and co_target_scalers is not None:
                # Scale inputs
                scaled_input = co_input_scaler.transform(prediction_data['co']).reshape(1, 24, 3)
                
                # Predict
                pred_scaled = co_model.predict(scaled_input)  # shape (1, 336, 1)
                print(f"CO prediction shape: {pred_scaled.shape}")
                # Handle different output shapes
                if pred_scaled.shape[1] != 336:
                    print(f"Warning: CO model output shape {pred_scaled.shape} is not (1, 336, 1)")
                    pred_scaled = pred_scaled.reshape(-1, 1)
                else:
                    pred_scaled = pred_scaled.reshape(336, 1)
                
                # Inverse transform output
                pred_inverse = co_target_scalers[0].inverse_transform(pred_scaled).flatten()
                
                # Get predictions for next 3 hours
                co_horizon = min(3, pred_inverse.shape[0])
                co_list = []
                for h in range(co_horizon):
                    co_list.append({"hour_ahead": h + 1, "CO": max(0, float(pred_inverse[h]))})
                results['co'] = co_list
                if co_list:
                    print("CO prediction using ML model (H+1)")
            else:
                # Fallback to trend analysis if model not loaded
                current_co = float(prediction_data['co'][-1][0])
                results['co'] = [
                    {"hour_ahead": 1, "CO": max(0, current_co * 1.05)}
                ]
                print("CO prediction using fallback (model not loaded)")
        except Exception as e:
            print(f"CO prediction error: {e}")
            results['co'] = [
                {"hour_ahead": 1, "CO": 0.0}  # Conservative fallback
            ]
        
        print("✅ All predictions completed successfully")
        
        # Add data freshness info
        data_freshness = "unknown"
        if sensor_data and len(sensor_data) > 0:
            latest_data_time = sensor_data[0]['timestamp']  # First item is newest
            data_freshness = latest_data_time
        
        return jsonify({
            "predictions": results,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_count": total_instances,
            "total_instances": total_instances,
            "data_freshness": data_freshness,
            "models_used": {
                "pm": pm_model is not None,
                "no2": no2_model is not None,
                "co": co_model is not None
            }
        })
        
    except Exception as e:
        print(f"❌ Critical error in predict_all: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "error_type": type(e).__name__
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Load models on startup
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
