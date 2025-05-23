# 🏭 Smart Factory Energy Prediction

![Project Banner](https://img.shields.io/badge/Status-Active-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-blue)  
![Python](https://img.shields.io/badge/Python-3.11-blue)  
![Render](https://img.shields.io/badge/Deployed%20on-Render-orange)

Welcome to the **Smart Factory Energy Prediction** project! This is an end-to-end MLOps pipeline for predicting energy consumption in a smart factory using machine learning. The project leverages an ElasticNet model to forecast energy usage based on environmental and operational data, such as temperature, humidity, and historical energy consumption. The app is deployed on Render and accessible live at:

🌐 [**Live Demo on Render**](https://smart-factory-energy-prediction.onrender.com/)

---

## 📖 Project Overview

This project demonstrates a complete MLOps workflow, from data ingestion to model deployment. It includes data preprocessing, model training with MLflow tracking, and a Flask-based web application for real-time and batch energy predictions. The pipeline was initially deployed on Google Kubernetes Engine (GKE) but, due to free-tier limitations on GCP, the cluster was deleted, and the app is now hosted on Render for seamless access.

### Key Features
- **Data Ingestion**: Downloads raw data from a public GitHub repository.
- **Data Preprocessing**: Handles missing values, outliers, and feature engineering (e.g., lag features, heat index).
- **Model Training**: Trains an ElasticNet model with performance metrics (R²: 0.9985, RMSE: 5.11, MAE: 2.13).
- **MLOps**: Uses MLflow for experiment tracking and CircleCI for CI/CD.
- **Web App**: A Flask app for single and batch predictions with a user-friendly UI.
- **Deployment**: Successfully deployed on Render (previously on GKE).

---

## 🗂️ Project Structure

Here’s the structure of the repository:

```
End-to-End-MLOps-Smart-Factory-Energy-Prediction
├── .circleci/                  # CircleCI configuration for CI/CD
│   └── config.yml
├── artifacts/                  # Directory for storing pipeline artifacts
│   ├── models/                # Trained models (e.g., ElasticNet_energy_model.pkl)
│   ├── processed/             # Processed data and scaler (e.g., scaler.pkl)
│   └── raw/                   # Raw data downloaded during ingestion
├── config/                     # Configuration files
│   ├── __pycache__/
│   ├── __init__.py
│   ├── config.yaml            # General configuration
│   └── paths_config.py        # File paths configuration
├── logs/                       # Log files (e.g., pipeline.log)
├── mlartifacts/                # MLflow artifacts
├── mlruns/                     # MLflow experiment tracking
├── models/                     # Model-related files (if any)
├── notebooks/                  # Jupyter notebooks for experimentation
├── pipeline/                   # Training pipeline scripts
│   ├── __init__.py
│   └── training_pipeline.py   # Main pipeline script
├── smart_factory_energy_prediction.egg-info/  # Package metadata
├── src/                        # Source code for the pipeline
│   ├── __pycache__/
│   ├── __init__.py
│   ├── custom_exception.py    # Custom exception handling
│   ├── data_ingestion.py      # Data ingestion logic
│   ├── data_processing.py     # Data preprocessing logic
│   ├── logger.py              # Logging utility
│   └── model_training.py      # Model training logic
├── static/                     # Static files for Flask app
│   ├── css/                   # CSS styles
│   └── templates/             # HTML templates
├── .gitignore                  # Git ignore file
├── app.py                      # Flask application for predictions
├── Dockerfile                  # Docker configuration for deployment
├── gcp-key.json                # GCP service account key (not tracked in Git)
├── kubernetes-deployment.yaml  # Kubernetes deployment and service config
├── LICENSE                     # MIT License file
├── mlflow.db                   # MLflow tracking database
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Python dependencies
└── setup.py                    # Setup script for package
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.11
- **Docker**: For containerization
- **Render Account**: For deployment (alternatively, GCP for Kubernetes deployment)
- **Git**: To clone the repository

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/End-to-End-MLOps-Smart-Factory-Energy-Prediction.git
   cd End-to-End-MLOps-Smart-Factory-Energy-Prediction
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Training Pipeline** (Optional):
   - This step downloads data, preprocesses it, trains the model, and saves artifacts.
   ```bash
   python pipeline/training_pipeline.py
   ```
   - Artifacts will be saved in `artifacts/` (e.g., `artifacts/models/ElasticNet_energy_model.pkl`).

5. **Run the Flask App Locally**:
   ```bash
   python app.py
   ```
   - Open `http://localhost:5000` in your browser to access the app.

---

## 🛠️ Deployment

### Local Deployment
- After running the Flask app (`python app.py`), you can access the app at `http://localhost:5000`.
- Use the UI to make single predictions or upload a CSV for batch predictions.

### Deployment on Render
The app is currently deployed on Render for easy access:

🌐 [**Live Demo on Render**](https://smart-factory-energy-prediction.onrender.com/)

**Steps to Deploy on Render**:
1. Create a new Web Service on Render.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
4. Deploy the app and access the provided URL.

### Previous Deployment on GKE
- The app was initially deployed on Google Kubernetes Engine (GKE) using CircleCI for CI/CD.
- Due to free-tier limitations on GCP, the cluster was deleted.
- The `kubernetes-deployment.yaml` file contains the Kubernetes configuration used for GKE deployment.

---

## 📊 Model Performance

The model was trained using an ElasticNet regression algorithm on a dataset with features like `energy_rolling_mean_3h`, `energy_lag_1h`, `zone3_temperature`, and `avg_zone_humidity`. The performance metrics are:

- **R² Score**: 0.9985
- **RMSE**: 5.11
- **MAE**: 2.13

Feature importance highlights the significance of historical energy consumption features:

| Feature               | Importance |
|-----------------------|------------|
| `energy_rolling_mean_3h` | 0.5265     |
| `energy_lag_2h`       | 0.2357     |
| `energy_lag_1h`       | 0.2341     |

---

## 🔄 CI/CD Pipeline

The project uses CircleCI for continuous integration and deployment. The pipeline includes:

- **Checkout Code**: Clones the repository.
- **Build Docker Image**: Builds and pushes the Docker image to Google Container Registry.
- **Deploy to GKE** (previously): Deploys the app to a GKE cluster (now replaced with Render deployment).

The configuration is defined in `.circleci/config.yml`.

---

## 📝 Usage

### Single Prediction
1. Open the app at [https://smart-factory-energy-prediction.onrender.com/](https://smart-factory-energy-prediction.onrender.com/).
2. Navigate to the "Single Prediction" tab.
3. Enter values for the 10 features (e.g., `energy_rolling_mean_3h=100`, `zone3_temperature=24`).
4. Click "Predict Energy Consumption" to get the predicted energy usage.

### Batch Prediction
1. Navigate to the "Batch Prediction" tab.
2. Upload a CSV file with the required features (template available in the app).
3. Download the predictions as a CSV file.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or feedback, feel free to reach out:

- **GitHub**: [GitHub Profile](https://github.com/FaheemKhan0817)
- **Email**: faheemthakur23@gmail.com

---

## 🙏 Acknowledgements

- **Dataset**: [Smart Factory Energy Dataset](https://raw.githubusercontent.com/FaheemKhan0817/Datasets-for-ML-Projects/refs/heads/main/Smart%20Factory%20Energy%20Dataset.csv)
- **Tools**: Flask, MLflow, CircleCI, Render, Kubernetes, GCP
- **Libraries**: Scikit-learn, Pandas, NumPy

Thank you for exploring the Smart Factory Energy Prediction project! 🚀