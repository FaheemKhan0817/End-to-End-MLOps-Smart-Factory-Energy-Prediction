import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.logger import get_logger
from src.custom_exception import CustomException
import joblib

class ModelTraining:
    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.logger = get_logger(__name__)
        self.logger.info("Model Training Initialized...")

    def train_elasticnet(self, X_train, X_test, y_train, y_test):
        try:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            self.logger.info("ElasticNet Model Performance:")
            self.logger.info(f"R²: {r2:.4f}")
            self.logger.info(f"RMSE: {rmse:.2f}")
            self.logger.info(f"MAE: {mae:.2f}")
            importance = np.abs(model.coef_)
            importance = importance / importance.sum() if importance.sum() > 0 else importance
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            self.logger.info("Feature Importance:")
            for i, row in feature_importance.iterrows():
                self.logger.info(f"{i+1}. {row['Feature']} - {row['Importance']:.4f}")
            model_path = os.path.join(self.output_path, 'ElasticNet_energy_model.pkl')
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved to {model_path}")
            feature_importance.to_csv(os.path.join(self.output_path, 'feature_importance.csv'), index=False)
            self.logger.info(f"Feature importance saved to {os.path.join(self.output_path, 'feature_importance.csv')}")
            joblib.dump(scaler, os.path.join(self.output_path, 'scaler.pkl'))
            self.logger.info(f"Scaler saved to {os.path.join(self.output_path, 'scaler.pkl')}")
            return model, y_pred, r2, rmse, mae, feature_importance
        except Exception as e:
            self.logger.error(f"Error while training ElasticNet: {e}")
            raise CustomException("Failed to train ElasticNet", str(e))

    def run(self, X_train, X_test, y_train, y_test):
        try:
            self.logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
            self.logger.info(f"Top 10 features used for training: {X_train.columns.tolist()}")
            model, y_pred, r2, rmse, mae, feature_importance = self.train_elasticnet(X_train, X_test, y_train, y_test)
            self.logger.info("Model Training pipeline executed successfully...")
            return model, y_pred, r2, rmse, mae, feature_importance
        except Exception as e:
            self.logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Failed to execute model training pipeline", str(e))

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessing
    input_path = r"C:\ML Projects\DS-Intern-Assignment-Faheem-Khan\data\data.csv"
    output_path = r"C:\MLOps Projects\End-to-End-MLOps-Smart-Factory-Energy-Prediction\artifacts\models"
    preprocessor = DataPreprocessing(input_path, output_path)
    X_train, X_test, y_train, y_test = preprocessor.run()
    trainer = ModelTraining(output_path)
    model, y_pred, r2, rmse, mae, feature_importance = trainer.run(X_train, X_test, y_train, y_test)