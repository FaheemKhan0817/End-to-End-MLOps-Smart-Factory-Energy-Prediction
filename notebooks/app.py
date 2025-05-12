import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained ElasticNet model and scaler
MODEL_PATH = "models/ElasticNet_energy_model_top_features.pkl"
SCALER_PATH = "models/ElasticNet_scaler_top_features.pkl"
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

# Set up the Streamlit app
st.set_page_config(
    page_title="Smart Factory Energy Prediction",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS to beautify the UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .reportview-container .main .block-container {
        max-width: 900px;
        padding: 1.5rem 1.5rem 1.5rem 1.5rem;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# App title
st.title("🏭 Smart Factory Energy Prediction")

# Sidebar
st.sidebar.header("Prediction Options")
prediction_mode = st.sidebar.radio(
    "Choose Prediction Mode:",
    ("Single Prediction", "Batch Prediction (CSV Upload)")
)

# Single Prediction Mode
if prediction_mode == "Single Prediction":
    st.subheader("Enter Sensor Data")
    st.write("Fill in the required fields to predict energy consumption:")

    # User inputs for each feature
    input_data = {}
    for feature in selected_features:
        if "temperature" in feature or "heat_index" in feature:
            input_data[feature] = st.slider(
                f"{feature.replace('_', ' ').capitalize()} (°C)",
                min_value=0.0, max_value=50.0, value=25.0
            )
        elif "energy" in feature:
            input_data[feature] = st.number_input(
                f"{feature.replace('_', ' ').capitalize()} (kWh)",
                min_value=0.0, max_value=1000.0, value=100.0
            )
        elif "humidity" in feature:
            input_data[feature] = st.slider(
                f"{feature.replace('_', ' ').capitalize()} (%)",
                min_value=0.0, max_value=100.0, value=50.0
            )

    # Predict button
    if st.button("Predict Energy Consumption"):
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess input data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Energy Consumption: **{prediction[0]:.2f} kWh**")

# Batch Prediction Mode
elif prediction_mode == "Batch Prediction (CSV Upload)":
    st.subheader("Upload CSV File")
    st.write("Upload a CSV file containing the sensor data for batch predictions:")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file:
        # Read the uploaded CSV
        batch_data = pd.read_csv(uploaded_file)

        # Validate the required columns
        missing_columns = [col for col in selected_features if col not in batch_data.columns]
        if missing_columns:
            st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
        else:
            # Preprocess the data using the loaded scaler
            batch_data_scaled = scaler.transform(batch_data[selected_features])

            # Make batch predictions
            batch_predictions = model.predict(batch_data_scaled)

            # Add predictions to the DataFrame
            batch_data["Predicted_Energy_Consumption"] = batch_predictions

            # Display the results
            st.write("Batch Predictions:")
            st.dataframe(batch_data)

            # Download button
            st.download_button(
                label="Download Predictions as CSV",
                data=batch_data.to_csv(index=False),
                file_name="predicted_energy_consumption.csv",
                mime="text/csv"
            )