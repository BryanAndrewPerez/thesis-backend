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
CORS(app, origins=['https://thesis-website-deployment.onrender.com'], methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], allow_headers=['Content-Type', 'Authorization'])


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
PM_MODEL_PATH = "models/chunk_pmonly.keras"
PM_INPUT_SCALER_PATH = "models/input_scaler_pmonly.pkl"
PM_TARGET_SCALERS_PATH = "models/target_scalers_pmonly.pkl"

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
    
    try:
        # Load PM model and scalers
        print(f"🔍 Checking PM model at: {PM_MODEL_PATH}")
        if os.path.exists(PM_MODEL_PATH):
            print("✅ PM model file found, loading...")
            pm_model = tf.keras.models.load_model(PM_MODEL_PATH)
            with open(PM_INPUT_SCALER_PATH, "rb") as f:
                pm_input_scaler = pickle.load(f)
            with open(PM_TARGET_SCALERS_PATH, "rb") as f:
                pm_target_scalers = pickle.load(f)
            print("✅ PM model loaded successfully")
        else:
            print(f"❌ PM model not found at {PM_MODEL_PATH}")
            
        # Load NO2 model and scalers
        print(f"🔍 Checking NO2 model at: {NO2_MODEL_PATH}")
        if os.path.exists(NO2_MODEL_PATH):
            print("✅ NO2 model file found, loading...")
            no2_model = tf.keras.models.load_model(NO2_MODEL_PATH)
            with open(NO2_INPUT_SCALER_PATH, "rb") as f:
                no2_input_scaler = pickle.load(f)
            with open(NO2_TARGET_SCALERS_PATH, "rb") as f:
                no2_target_scalers = pickle.load(f)
            print("✅ NO2 model loaded successfully")
        else:
            print(f"❌ NO2 model not found at {NO2_MODEL_PATH}")
            
        # Load CO model and scalers
        print(f"🔍 Checking CO model at: {CO_MODEL_PATH}")
        if os.path.exists(CO_MODEL_PATH):
            print("✅ CO model file found, loading...")
            co_model = tf.keras.models.load_model(CO_MODEL_PATH)
            with open(CO_INPUT_SCALER_PATH, "rb") as f:
                co_input_scaler = pickle.load(f)
            with open(CO_TARGET_SCALERS_PATH, "rb") as f:
                co_target_scalers = pickle.load(f)
            print("✅ CO model loaded successfully")
        else:
            print(f"❌ CO model not found at {CO_MODEL_PATH}")
            
        # Summary
        print("📊 Model Loading Summary:")
        print(f"   PM Model: {'✅ Loaded' if pm_model is not None else '❌ Failed'}")
        print(f"   NO2 Model: {'✅ Loaded' if no2_model is not None else '❌ Failed'}")
        print(f"   CO Model: {'✅ Loaded' if co_model is not None else '❌ Failed'}")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
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
        # Add cache-busting to ensure fresh data
        ref = db.reference('/sensors')
        if location:
            ref = ref.child(location)
            
        # Force fresh data fetch
        data = ref.get()
        
        print(f"Firebase data fetched: {type(data)}")
        if data:
            print(f"Data keys: {list(data.keys())[:5] if isinstance(data, dict) else 'Not a dict'}")
            
            # Check field names in the first reading
            if isinstance(data, dict):
                first_device = list(data.values())[0]
                if isinstance(first_device, dict):
                    first_reading = list(first_device.values())[0]
                    if isinstance(first_reading, dict):
                        field_names = list(first_reading.keys())
                        print(f"Available field names: {field_names}")
                        pm_fields = [f for f in field_names if 'pm' in f.lower()]
                        print(f"PM-related fields: {pm_fields}")
        else:
            print("No data found in Firebase")
        
        if not data:
            return None
            
        # Process all data from Firebase
        processed_data = []
        
        # Handle the nested structure: device_id -> timestamp_key -> readings
        # Process data in reverse order to get newest first
        all_readings = []
        for device_id, device_data in data.items():
            if isinstance(device_data, dict):
                for timestamp_key, readings in device_data.items():
                    all_readings.append({
                        'device_id': device_id,
                        'timestamp_key': timestamp_key,
                        'readings': readings
                    })
        
        # Sort by timestamp_key (Firebase push keys are chronological)
        all_readings.sort(key=lambda x: x['timestamp_key'])
        
        # Process the sorted readings
        for reading in all_readings:
            try:
                readings = reading['readings']
                # Use current time minus offset to create chronological order
                data_time = datetime.now() - timedelta(hours=len(processed_data))
                
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
                    print(f"Reading {len(processed_data)}: PM fields found: {pm_fields}, using value: {pm25_value}")
                
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
                print(f"Error processing reading {reading['timestamp_key']}: {e}")
                continue
                
        # Sort by timestamp and get the most recent data
        processed_data.sort(key=lambda x: x['time'])
        print(f"Processed {len(processed_data)} data points from Firebase")
        if processed_data:
            print(f"Oldest data: {processed_data[0]['timestamp']}")
            print(f"Newest data: {processed_data[-1]['timestamp']}")
            print(f"Total instances available: {len(processed_data)}")
        
        # Return the most recent 24 instances for prediction
        recent_data = processed_data[-24:] if len(processed_data) >= 24 else processed_data
        print(f"Using {len(recent_data)} most recent instances for prediction")
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

@app.route("/reload_models", methods=["POST"])
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
            }
        })
    except Exception as e:
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
        location = data.get("location", "")
        
        # Fetch data from Firebase
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
        
        # Lazy-load models on first use if not already loaded
        global pm_model, no2_model, co_model
        if pm_model is None or no2_model is None or co_model is None:
            print("⏳ Models not loaded yet. Attempting lazy load...")
            load_models()
            print(f"Models loaded status after lazy load: pm={pm_model is not None}, no2={no2_model is not None}, co={co_model is not None}")

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
        
        location = data.get("location", "")
        print(f"🔍 Location: {location}")
        
        # Fetch data from Firebase
        print("🔍 Fetching Firebase data...")
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
        print(f"   First 5 instances:")
        for i in range(min(5, len(prediction_data['pm']))):
            row = prediction_data['pm'][i]
            print(f"     [{i}]: PM2.5={row[0]:.2f}, PM10={row[1]:.2f}, sin={row[2]:.3f}, cos={row[3]:.3f}")
        print(f"   Last 5 instances:")
        for i in range(max(0, len(prediction_data['pm'])-5), len(prediction_data['pm'])):
            row = prediction_data['pm'][i]
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
        return jsonify({
            "predictions": results,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_count": total_instances,
            "total_instances": total_instances
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
