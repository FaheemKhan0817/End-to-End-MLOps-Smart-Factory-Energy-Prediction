from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import os

app = Flask(__name__)

# Load the trained ElasticNet model and scaler
MODEL_PATH = "artifacts\models\ElasticNet_energy_model_top_features.pkl"
SCALER_PATH = "artifacts\models\ElasticNet_scaler_top_features.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define the list of input features
selected_features = [
    'energy_rolling_mean_3h',
    'energy_lag_2h',
    'energy_lag_1h',
    'zone3_heat_index',
    'zone3_temperature',
    'avg_zone_humidity',
    'zone1_heat_index',
    'zone1_temperature',
    'zone2_temperature',
    'zone6_temperature'
]

@app.route('/')
def home():
    return render_template('index.html', selected_features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {}
        for feature in selected_features:
            value = request.form.get(feature)
            if not value:
                return render_template('index.html', selected_features=selected_features, 
                                     error=f"Missing value for {feature}")
            input_data[feature] = float(value)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale inputs
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return render_template('index.html', selected_features=selected_features, 
                             prediction=f"{prediction:.2f} kWh")
    except Exception as e:
        return render_template('index.html', selected_features=selected_features, 
                             error=f"Error: {str(e)}")

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', selected_features=selected_features, 
                                 error="No file uploaded")
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return render_template('index.html', selected_features=selected_features, 
                                 error="Please upload a CSV file")

        # Read CSV
        batch_data = pd.read_csv(file)

        # Validate columns
        missing_columns = [col for col in selected_features if col not in batch_data.columns]
        if missing_columns:
            return render_template('index.html', selected_features=selected_features, 
                                 error=f"Missing columns: {', '.join(missing_columns)}")

        # Scale and predict
        batch_data_scaled = scaler.transform(batch_data[selected_features])
        batch_predictions = model.predict(batch_data_scaled)
        batch_data['Predicted_Energy_Consumption'] = batch_predictions

        # Convert to CSV for download
        output = StringIO()
        batch_data.to_csv(output, index=False)
        output.seek(0)

        # Save to temporary file for download
        temp_file = "predicted_energy_consumption.csv"
        batch_data.to_csv(temp_file, index=False)

        # Render results table
        return render_template('index.html', selected_features=selected_features, 
                             batch_results=batch_data.to_dict(orient='records'), 
                             download_file=temp_file)
    except Exception as e:
        return render_template('index.html', selected_features=selected_features, 
                             error=f"Error: {str(e)}")

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return render_template('index.html', selected_features=selected_features, 
                             error=f"Error downloading file: {str(e)}")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)