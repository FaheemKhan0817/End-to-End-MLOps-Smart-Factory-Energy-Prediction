import os

# Directory for raw data
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

# Directory for processed data
PROCESSED_DIR = "artifacts/processed"
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.pkl")

# Directory for model artifacts
MODEL_DIR = "artifacts/models"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "ElasticNet_energy_model.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")

# Configuration file
CONFIG_PATH = "config/config.yaml"
