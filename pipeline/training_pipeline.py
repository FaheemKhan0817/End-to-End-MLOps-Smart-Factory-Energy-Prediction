import os
import joblib
from src.data_ingestion import DataIngestion
from src.data_processing import DataPreprocessing
from src.model_training import ModelTraining
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_FILE_PATH, PROCESSED_DIR, MODEL_DIR, SCALER_PATH, MODEL_FILE_PATH, FEATURE_IMPORTANCE_PATH

if __name__ == "__main__":
    try:
        logger = get_logger(__name__)
        logger.info("Initiating training pipeline")

        # Verify MLflow server
        logger.info("Checking MLflow server availability")
        try:
            import requests
            response = requests.get("http://localhost:5000", timeout=5)
            if response.status_code != 200:
                raise CustomException("MLflow server not running", "Response code: {}".format(response.status_code))
        except Exception as e:
            logger.error("MLflow server check failed: {}".format(str(e)))
            raise CustomException("MLflow server unavailable", str(e))

        # Data Ingestion
        logger.info("Starting data ingestion")
        data_ingestion = DataIngestion()
        df = data_ingestion.ingest_data()
        logger.info("Data ingestion completed successfully")

        # Data Preprocessing
        logger.info("Starting data preprocessing")
        preprocessor = DataPreprocessing(RAW_FILE_PATH, PROCESSED_DIR)
        X_train, X_test, y_train, y_test = preprocessor.run()
        logger.info("Data preprocessing completed successfully")

        # Verify scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler features: {}".format(scaler.feature_names_in_.tolist()))
        else:
            logger.error("Scaler file not found at {}".format(SCALER_PATH))
            raise CustomException("Scaler file missing", "Path: {}".format(SCALER_PATH))

        # Model Training
        logger.info("Starting model training")
        trainer = ModelTraining()
        model, y_pred, r2, rmse, mae, feature_importance = trainer.run(X_train, X_test, y_train, y_test)
        logger.info("Model training completed successfully")

        # Verify artifacts
        for artifact in [MODEL_FILE_PATH, FEATURE_IMPORTANCE_PATH]:
            if not os.path.exists(artifact):
                logger.error("Artifact not found: {}".format(artifact))
                raise CustomException("Artifact missing", "Path: {}".format(artifact))

        logger.info("Training pipeline executed successfully")
        logger.info("Pipeline artifacts saved: model={}, scaler={}, feature_importance={}".format(
            MODEL_FILE_PATH, SCALER_PATH, FEATURE_IMPORTANCE_PATH))

    except CustomException as ce:
        logger.error("CustomException in training pipeline: {}".format(str(ce)))
        raise
    except Exception as e:
        logger.error("Unexpected error in training pipeline: {}".format(str(e)))
        raise CustomException("Training pipeline failed", str(e))