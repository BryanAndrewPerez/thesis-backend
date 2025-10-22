from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
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
CORS(app, origins=["*"], methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

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
    """Load all ML models and scalers with better error handling"""
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

# Initialize Firebase with error handling
try:
    firebase_initialized = initialize_firebase()
except Exception as e:
    print(f"❌ Critical error during Firebase initialization: {e}")
    firebase_initialized = False

# Load models on startup with error handling
try:
    print("🔄 Loading ML models on startup...")
    load_models()
except Exception as e:
    print(f"❌ Critical error during model loading: {e}")
    import traceback
    traceback.print_exc()

def fetch_firebase_data(location="", hours=24):
    """Fetch sensor data from Firebase for the last N hours with better error handling"""
    print(f"🔍 fetch_firebase_data called with location='{location}', hours={hours}")
    print(f"🔍 firebase_initialized: {firebase_initialized}")
    
    if not firebase_initialized:
        print("❌ Firebase not initialized - returning None")
        return None
        
    try:
        # Fetch data from Firebase - using 'sensors' collection
        ref = db.reference('/sensors')
        if location:
            ref = ref.child(location)
            
        # Force fresh data fetch
        data = ref.get()
        
        print(f"Firebase data fetched: {type(data)}")
        if data:
            print(f"Data keys: {list(data.keys())[:5] if isinstance(data, dict) else 'Not a dict'}")
        else:
            print("No data found in Firebase")
        
        if not data:
            return None
            
        # Process all data from Firebase
        processed_data = []
        
        # Handle the nested structure: device_id -> timestamp_key -> readings
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
                
                # Handle PM2.5 field name variations
                pm25_value = (readings.get('PM2.5') or 
                             readings.get('pm2.5') or 
                             readings.get('pm2_5') or 
                             readings.get('pm25') or 
                             0)
                
                processed_data.append({
                    'timestamp': data_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'time': data_time,
                    'pm25': pm25_value,
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
        
        # Return the most recent 24 instances for prediction
        recent_data = processed_data[-24:] if len(processed_data) >= 24 else processed_data
        print(f"Using {len(recent_data)} most recent instances for prediction")
        return recent_data
        
    except Exception as e:
        print(f"Error fetching Firebase data: {e}")
        return None

def prepare_prediction_data(sensor_data):
    """Prepare sensor data for prediction models with error handling"""
    if not sensor_data or len(sensor_data) < 24:
        return None
        
    try:
        # Get the last 24 hours of data
        last_24_hours = sensor_data[-24:]
        
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
            raw_pm25 = float(data_point.get('pm25', 0))
            raw_pm10 = float(data_point.get('pm10', 0))
            scaled_pm25 = raw_pm25 * 2.0
            scaled_pm10 = raw_pm10 * 2.0
            pm_data.append([
                scaled_pm25,
                scaled_pm10,
                hour_sin,
                hour_cos
            ])
            
            # NO2 data: NO2, hour_sin, hour_cos
            raw_no2 = float(data_point.get('no2', 0))
            scaled_no2 = raw_no2 / 4.0
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
    except Exception as e:
        print(f"Error preparing prediction data: {e}")
        return None

@app.route("/", methods=["GET"])
def home():
    try:
        return jsonify({
            "message": "Air Quality Prediction API",
            "status": "running",
            "available_endpoints": {
                "/predict_pm": "POST - Predict PM2.5 and PM10 (requires 24x4 input: PM2.5, PM10, hour_sin, hour_cos)",
                "/predict_no2": "POST - Predict NO2 (requires 24x3 input: NO2, hour_sin, hour_cos)",
                "/predict_co": "POST - Predict CO (requires 24x3 input: CO, hour_sin, hour_cos)",
                "/predict_all": "GET/POST - Predict all pollutants using Firebase data",
                "/health": "GET - Check API health and model status",
                "/debug_firebase": "GET - Debug Firebase data structure and connection",
                "/test_predict_all": "POST - Test predict_all endpoint",
                "/simple_health": "GET - Simple health check (no dependencies)"
            }
        })
    except Exception as e:
        return jsonify({"error": f"Home endpoint error: {str(e)}"}), 500

@app.route("/simple_health", methods=["GET"])
def simple_health():
    """Simple health check that doesn't depend on Firebase or models"""
    try:
        return jsonify({
            "status": "healthy",
            "message": "Backend is running",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": f"Simple health check error: {str(e)}"}), 500

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
        
        # Test with minimal request
        data = request.json or {}
        location = data.get("location", "")
        
        return jsonify({
            "status": "test_successful",
            "models_status": models_status,
            "firebase_initialized": firebase_initialized,
            "location": location,
            "message": "Test endpoint working - try /predict_all for full prediction"
        })
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

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

@app.route("/predict_all", methods=["GET", "POST"])
def predict_all():
    """Predict all pollutants using Firebase data with comprehensive error handling"""
    try:
        print("🔍 predict_all endpoint called")
        
        # Handle both GET and POST requests
        if request.method == "GET":
            # For GET requests, use default values
            data = {}
            location = ""
            print("🔍 GET request - using default values")
        else:
            # For POST requests, check for JSON data
            if not request.json:
                print("❌ No JSON data in request")
                return jsonify({"error": "No JSON data provided"}), 400
                
            data = request.json
            location = data.get("location", "")
            print(f"🔍 POST request - Location: {location}")
        
        # Check if Firebase is initialized
        if not firebase_initialized:
            print("❌ Firebase not initialized")
            return jsonify({
                "error": "Firebase not initialized",
                "message": "Please check Firebase configuration",
                "firebase_initialized": False
            }), 500
        
        # Fetch data from Firebase
        print("🔍 Fetching Firebase data...")
        sensor_data = fetch_firebase_data(location)
        if not sensor_data:
            print("❌ No sensor data from Firebase")
            return jsonify({"error": "No sensor data available in Firebase"}), 400
            
        print(f"✅ Got {len(sensor_data)} data points from Firebase")
        
        # Prepare data for prediction
        print("🔍 Preparing prediction data...")
        prediction_data = prepare_prediction_data(sensor_data)
        if not prediction_data:
            print("❌ Failed to prepare prediction data")
            return jsonify({"error": "Insufficient data for prediction"}), 400
            
        print("✅ Prediction data prepared successfully")
        
        # Initialize results
        results = {}
        total_instances = len(sensor_data)
        print(f"🔍 Total instances: {total_instances}")
        print(f"🔍 Prediction data keys: {list(prediction_data.keys())}")
        
        # Check if any models are loaded
        models_available = {
            "pm": pm_model is not None and pm_input_scaler is not None and pm_target_scalers is not None,
            "no2": no2_model is not None and no2_input_scaler is not None and no2_target_scalers is not None,
            "co": co_model is not None and co_input_scaler is not None and co_target_scalers is not None
        }
        print(f"🔍 Models available: {models_available}")
        
        # Predict PM using ML model with error handling
        print("🔍 Starting PM prediction...")
        try:
            if pm_model is not None and pm_input_scaler is not None and pm_target_scalers is not None:
                # Scale inputs
                scaled_input = pm_input_scaler.transform(prediction_data['pm']).reshape(1, 24, 4)
                
                # Predict
                pred_scaled = pm_model.predict(scaled_input)
                
                # Handle different output shapes
                if pred_scaled.shape[1] != 336:
                    pred_scaled = pred_scaled.reshape(-1, 2)
                else:
                    pred_scaled = pred_scaled.reshape(336, 2)
                
                # Inverse transform each output column
                pred_inverse_cols = []
                for i, sc in enumerate(pm_target_scalers):
                    pred_inverse_cols.append(sc.inverse_transform(pred_scaled[:, i].reshape(-1, 1)).flatten())
                pred_inverse = np.vstack(pred_inverse_cols).T
                
                # Get predictions for next 3 hours and scale back down
                pm_horizon = min(3, pred_inverse.shape[0])
                pm_list = []
                for h in range(pm_horizon):
                    scaled_pm25 = float(pred_inverse[h][0]) / 2.0
                    scaled_pm10 = float(pred_inverse[h][1]) / 2.0
                    pm_list.append({"hour_ahead": h + 1, "PM2.5": max(0, scaled_pm25), "PM10": max(0, scaled_pm10)})
                results['pm'] = pm_list
                print(f"✅ PM prediction completed: {len(pm_list)} hours")
            else:
                # Fallback predictions
                results['pm'] = [
                    {"hour_ahead": 1, "PM2.5": 25.0, "PM10": 50.0},
                    {"hour_ahead": 2, "PM2.5": 25.0, "PM10": 50.0},
                    {"hour_ahead": 3, "PM2.5": 25.0, "PM10": 50.0}
                ]
                print("⚠️ PM prediction using fallback (model not loaded)")
        except Exception as e:
            print(f"❌ PM prediction error: {e}")
            results['pm'] = [
                {"hour_ahead": 1, "PM2.5": 25.0, "PM10": 50.0},
                {"hour_ahead": 2, "PM2.5": 25.0, "PM10": 50.0},
                {"hour_ahead": 3, "PM2.5": 25.0, "PM10": 50.0}
            ]
        
        # Predict NO2 using ML model with error handling
        print("🔍 Starting NO2 prediction...")
        try:
            if no2_model is not None and no2_input_scaler is not None and no2_target_scalers is not None:
                # Scale inputs
                scaled_input = no2_input_scaler.transform(prediction_data['no2']).reshape(1, 24, 3)
                
                # Predict
                pred_scaled = no2_model.predict(scaled_input)
                
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
                print(f"✅ NO2 prediction completed: {len(no2_list)} hours")
            else:
                # Fallback predictions
                results['no2'] = [
                    {"hour_ahead": 1, "NO2": 15.0},
                    {"hour_ahead": 2, "NO2": 15.0},
                    {"hour_ahead": 3, "NO2": 15.0}
                ]
                print("⚠️ NO2 prediction using fallback (model not loaded)")
        except Exception as e:
            print(f"❌ NO2 prediction error: {e}")
            results['no2'] = [
                {"hour_ahead": 1, "NO2": 15.0},
                {"hour_ahead": 2, "NO2": 15.0},
                {"hour_ahead": 3, "NO2": 15.0}
            ]
        
        # Predict CO using ML model with error handling
        print("🔍 Starting CO prediction...")
        try:
            if co_model is not None and co_input_scaler is not None and co_target_scalers is not None:
                # Scale inputs
                scaled_input = co_input_scaler.transform(prediction_data['co']).reshape(1, 24, 3)
                
                # Predict
                pred_scaled = co_model.predict(scaled_input)
                
                # Handle different output shapes
                if pred_scaled.shape[1] != 336:
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
                print(f"✅ CO prediction completed: {len(co_list)} hours")
            else:
                # Fallback predictions
                results['co'] = [
                    {"hour_ahead": 1, "CO": 0.0},
                    {"hour_ahead": 2, "CO": 0.0},
                    {"hour_ahead": 3, "CO": 0.0}
                ]
                print("⚠️ CO prediction using fallback (model not loaded)")
        except Exception as e:
            print(f"❌ CO prediction error: {e}")
            results['co'] = [
                {"hour_ahead": 1, "CO": 0.0},
                {"hour_ahead": 2, "CO": 0.0},
                {"hour_ahead": 3, "CO": 0.0}
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
