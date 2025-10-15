# Air Quality Prediction Backend

This Flask backend provides prediction services for air quality parameters (PM2.5, PM10, NO2, CO) using machine learning models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Firebase:
   - Download your Firebase service account key
   - Place it in the backend directory
   - Update the path in `config.py` or set the `FIREBASE_SERVICE_ACCOUNT_KEY_PATH` environment variable

3. Place your ML models in the `models/` directory:
   - `chunk_pmonly.keras`
   - `input_scaler_pmonly.pkl`
   - `target_scalers_pmonly.pkl`
   - `chunk_no2only.keras`
   - `input_scaler_no2only.pkl`
   - `target_scalers_no2only.pkl`
   - `chunk_coonly (1).keras`
   - `input_scaler_coonly.pkl`
   - `target_scalers_coonly.pkl`

4. Run the application:
```bash
python app.py
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check and model status
- `POST /predict_pm` - Predict PM2.5 and PM10
- `POST /predict_no2` - Predict NO2
- `POST /predict_co` - Predict CO
- `POST /predict_all` - Predict all pollutants using Firebase data

## Usage

The `/predict_all` endpoint automatically fetches the last 24 hours of sensor data from Firebase and generates predictions for all pollutants.
