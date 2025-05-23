import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, send_file
from io import StringIO
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import MODEL_FILE_PATH, SCALER_PATH

app = Flask(__name__)

# Set up logger
logger = get_logger(__name__)

# Load the trained ElasticNet model and scaler
try:
    model = joblib.load(MODEL_FILE_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Model loaded from {MODEL_FILE_PATH}")
    logger.info(f"Scaler loaded from {SCALER_PATH}")
    logger.info(f"Scaler features: {scaler.feature_names_in_.tolist()}")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {str(e)}")
    raise CustomException("Model or scaler loading failed", str(e))

# Define the list of input features
selected_features = [
    'energy_rolling_mean_3h', 'energy_lag_2h', 'energy_lag_1h', 'zone3_heat_index',
    'zone3_temperature', 'avg_zone_humidity', 'zone1_heat_index', 'zone1_temperature',
    'zone2_temperature', 'zone6_temperature'
]

# Define feature constraints
FEATURE_CONSTRAINTS = {
    'energy_rolling_mean_3h': {'min': 0, 'max': 1000, 'unit': 'kWh'},
    'energy_lag_2h': {'min': 0, 'max': 1000, 'unit': 'kWh'},
    'energy_lag_1h': {'min': 0, 'max': 1000, 'unit': 'kWh'},
    'zone3_heat_index': {'min': 0, 'max': 50, 'unit': '°C'},
    'zone3_temperature': {'min': 0, 'max': 50, 'unit': '°C'},
    'avg_zone_humidity': {'min': 0, 'max': 100, 'unit': '%'},
    'zone1_heat_index': {'min': 0, 'max': 50, 'unit': '°C'},
    'zone1_temperature': {'min': 0, 'max': 50, 'unit': '°C'},
    'zone2_temperature': {'min': 0, 'max': 50, 'unit': '°C'},
    'zone6_temperature': {'min': 0, 'max': 50, 'unit': '°C'}
}

@app.route('/')
def home():
    logger.info("Rendering home page")
    return render_template('index.html', selected_features=selected_features, constraints=FEATURE_CONSTRAINTS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received single prediction request")
        input_data = {}
        for feature in selected_features:
            value = request.form.get(feature)
            if not value:
                logger.warning(f"Missing value for {feature}")
                return render_template('index.html', selected_features=selected_features, 
                                     constraints=FEATURE_CONSTRAINTS, error=f"Missing value for {feature}")
            try:
                value = float(value)
                # Validate constraints
                constraints = FEATURE_CONSTRAINTS[feature]
                if value < constraints['min'] or value > constraints['max']:
                    logger.warning(f"Invalid value for {feature}: {value}")
                    return render_template('index.html', selected_features=selected_features, 
                                         constraints=FEATURE_CONSTRAINTS, 
                                         error=f"{feature} must be between {constraints['min']} and {constraints['max']} {constraints['unit']}")
                input_data[feature] = value
            except ValueError:
                logger.warning(f"Invalid input for {feature}: {value}")
                return render_template('index.html', selected_features=selected_features, 
                                     constraints=FEATURE_CONSTRAINTS, error=f"Invalid input for {feature}")
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        if prediction < 0:
            logger.warning("Negative prediction detected, clipping to 0")
            prediction = 0
        logger.info(f"Prediction successful: {prediction:.2f} kWh")
        return render_template('index.html', selected_features=selected_features, 
                             constraints=FEATURE_CONSTRAINTS, prediction=f"{prediction:.2f} kWh")
    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}")
        return render_template('index.html', selected_features=selected_features, 
                             constraints=FEATURE_CONSTRAINTS, error=f"Error: {str(e)}")

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        logger.info("Received batch prediction request")
        if 'file' not in request.files:
            logger.warning("No file uploaded")
            return render_template('index.html', selected_features=selected_features, 
                                 constraints=FEATURE_CONSTRAINTS, error="No file uploaded")
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            logger.warning("Invalid file type uploaded")
            return render_template('index.html', selected_features=selected_features, 
                                 constraints=FEATURE_CONSTRAINTS, error="Please upload a CSV file")
        batch_data = pd.read_csv(file)
        missing_columns = [col for col in selected_features if col not in batch_data.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {', '.join(missing_columns)}")
            return render_template('index.html', selected_features=selected_features, 
                                 constraints=FEATURE_CONSTRAINTS, 
                                 error=f"Missing columns: {', '.join(missing_columns)}")
        # Validate batch data
        for feature in selected_features:
            constraints = FEATURE_CONSTRAINTS[feature]
            invalid_rows = batch_data[
                (batch_data[feature] < constraints['min']) | (batch_data[feature] > constraints['max'])
            ]
            if not invalid_rows.empty:
                logger.warning(f"Invalid values in {feature} for {len(invalid_rows)} rows")
                return render_template('index.html', selected_features=selected_features, 
                                     constraints=FEATURE_CONSTRAINTS, 
                                     error=f"Invalid values in {feature} must be between {constraints['min']} and {constraints['max']} {constraints['unit']}")
        batch_data_scaled = scaler.transform(batch_data[selected_features])
        batch_predictions = model.predict(batch_data_scaled)
        batch_predictions = np.clip(batch_predictions, 0, None)  # Clip negative predictions
        batch_data['Predicted_Energy_Consumption'] = batch_predictions
        output = StringIO()
        batch_data.to_csv(output, index=False)
        output.seek(0)
        temp_file = "predicted_energy_consumption.csv"
        batch_data.to_csv(temp_file, index=False)
        logger.info("Batch prediction successful")
        return render_template('index.html', selected_features=selected_features, 
                             constraints=FEATURE_CONSTRAINTS, 
                             batch_results=batch_data.to_dict(orient='records'), 
                             download_file=temp_file)
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return render_template('index.html', selected_features=selected_features, 
                             constraints=FEATURE_CONSTRAINTS, error=f"Error: {str(e)}")

@app.route('/download/<filename>')
def download_file(filename):
    try:
        logger.info(f"Downloading file: {filename}")
        return send_file(filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return render_template('index.html', selected_features=selected_features, 
                             constraints=FEATURE_CONSTRAINTS, error=f"Error downloading file: {str(e)}")

if __name__ == '__main__':
    os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=5000, debug=True)