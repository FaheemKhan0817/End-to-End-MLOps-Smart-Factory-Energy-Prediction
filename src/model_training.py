import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.logger import get_logger
from src.custom_exception import CustomException
import joblib
import mlflow
import mlflow.sklearn
from config.paths_config import MODEL_DIR, MODEL_FILE_PATH, FEATURE_IMPORTANCE_PATH

class ModelTraining:
    def __init__(self):
        self.model_dir = MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger = get_logger(__name__)
        self.logger.info("Model Training Initialized...")
        mlflow.set_tracking_uri("http://localhost:5000")
        self.logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
        # Ensure experiment exists
        self.experiment_name = "SmartFactoryEnergyPrediction"
        if not mlflow.get_experiment_by_name(self.experiment_name):
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
        self.logger.info(f"MLflow experiment set to: {self.experiment_name}")

    def train_elasticnet(self, X_train, X_test, y_train, y_test):
        try:
            with mlflow.start_run():
                # Use pre-scaled data from data_processing.py
                self.logger.info("Using pre-scaled training and test data")

                # Define and train model
                params = {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 10000, "random_state": 42}
                model = ElasticNet(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                self.logger.info("ElasticNet Model Performance:")
                self.logger.info(f"R²: {r2:.4f}")
                self.logger.info(f"RMSE: {rmse:.2f}")
                self.logger.info(f"MAE: {mae:.2f}")

                # Compute feature importance
                importance = np.abs(model.coef_)
                importance = importance / importance.sum() if importance.sum() > 0 else importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                self.logger.info("Feature Importance:")
                for i, row in feature_importance.iterrows():
                    self.logger.info(f"{i+1}. {row['Feature']} - {row['Importance']:.4f}")

                # Save artifacts locally
                joblib.dump(model, MODEL_FILE_PATH)
                self.logger.info(f"Model saved to {MODEL_FILE_PATH}")
                feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
                self.logger.info(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")

                # Log parameters, metrics, and artifacts to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics({"R2": r2, "RMSE": rmse, "MAE": mae})
                mlflow.log_artifact(FEATURE_IMPORTANCE_PATH)
                mlflow.sklearn.log_model(model, "elasticnet_model", registered_model_name="ElasticNetEnergyModel")
                self.logger.info(f"Model and artifacts logged to MLflow run {mlflow.active_run().info.run_id}")

                return model, y_pred, r2, rmse, mae, feature_importance
        except Exception as e:
            self.logger.error(f"Error while training ElasticNet: {str(e)}")
            raise CustomException("Failed to train ElasticNet", str(e))

    def run(self, X_train, X_test, y_train, y_test):
        try:
            self.logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
            self.logger.info(f"Top 10 features used for training: {X_train.columns.tolist()}")
            model, y_pred, r2, rmse, mae, feature_importance = self.train_elasticnet(X_train, X_test, y_train, y_test)
            self.logger.info("Model Training pipeline executed successfully...")
            return model, y_pred, r2, rmse, mae, feature_importance
        except Exception as e:
            self.logger.error(f"Error in model training pipeline: {str(e)}")
            raise CustomException("Failed to execute model training pipeline", str(e))

if __name__ == "__main__":
    from src.data_processing import DataPreprocessing
    from config.paths_config import RAW_FILE_PATH, PROCESSED_DIR
    preprocessor = DataPreprocessing(RAW_FILE_PATH, PROCESSED_DIR)
    X_train, X_test, y_train, y_test = preprocessor.run()
    trainer = ModelTraining()
    model, y_pred, r2, rmse, mae, feature_importance = trainer.run(X_train, X_test, y_train, y_test)