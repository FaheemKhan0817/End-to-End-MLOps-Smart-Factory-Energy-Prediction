import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.logger import get_logger
from src.custom_exception import CustomException

# Configure logger
logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []

        os.makedirs(output_path, exist_ok=True)
        logger.info("Data Preprocessing Initialized...")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", str(e))

    def preprocess_data(self):
        try:
            # Validate timestamp column
            if 'timestamp' not in self.df.columns:
                raise ValueError("Timestamp column not found in DataFrame")

            # Convert data types and handle missing values
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            if self.df['timestamp'].isna().any():
                raise ValueError("Invalid timestamps found in DataFrame")

            for col in self.df.select_dtypes(include=['object']).columns:
                if col != 'timestamp':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    if self.df[col].isna().sum() > 0:
                        logger.warning(f"Non-numeric values in {col} converted to NaN")

            self.df = self.df.sort_values('timestamp').reset_index(drop=True)

            # Handle missing values
            # Drop rows with missing target values
            self.df = self.df.dropna(subset=['equipment_energy_consumption'])

            # Set timestamp as index for time-based interpolation
            self.df.set_index('timestamp', inplace=True)

            # Interpolate sensor columns (temperature, humidity, pressure)
            sensor_cols = [col for col in self.df.columns if any(x in col for x in ['temperature', 'humidity', 'pressure'])]
            if sensor_cols:
                try:
                    self.df[sensor_cols] = self.df[sensor_cols].interpolate(method='time', limit_direction='both')
                except NotImplementedError:
                    logger.warning("Time-based interpolation failed, falling back to linear interpolation")
                    self.df[sensor_cols] = self.df[sensor_cols].interpolate(method='linear', limit_direction='both')

            # Reset index
            self.df.reset_index(inplace=True)

            # For remaining columns, use median imputation
            remaining_cols = [col for col in self.df.columns if col not in sensor_cols and col != 'timestamp' and col != 'equipment_energy_consumption']
            for col in remaining_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())

            # Fix implausible values
            temp_cols = [col for col in self.df.columns if 'temperature' in col]
            for col in temp_cols:
                if 'outdoor' not in col:  # Indoor temperatures shouldn't be negative
                    neg_count = (self.df[col] < 0).sum()
                    if neg_count > 0:
                        logger.info(f"Fixing {neg_count} negative values in {col}")
                        median_val = self.df[self.df[col] > 0][col].median()
                        self.df.loc[self.df[col] < 0, col] = median_val if not pd.isna(median_val) else 0

            humidity_cols = [col for col in self.df.columns if 'humidity' in col]
            for col in humidity_cols:
                neg_count = (self.df[col] < 0).sum()
                high_count = (self.df[col] > 100).sum()
                if neg_count > 0 or high_count > 0:
                    logger.info(f"Fixing {neg_count} negative and {high_count} >100% values in {col}")
                    self.df.loc[self.df[col] < 0, col] = 0
                    self.df.loc[self.df[col] > 100, col] = 100

            # Drop specific columns with persistent missing values
            self.df = self.df.dropna(subset=['zone3_humidity', 'zone6_temperature'])

            # Drop duplicates
            self.df = self.df.drop_duplicates()

            logger.info("Basic preprocessing done.")
        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise CustomException("Failed to preprocess data", str(e))

    def feature_engineering(self):
        try:
            # Create time features
            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['day'] = self.df['timestamp'].dt.day
            self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
            self.df['month'] = self.df['timestamp'].dt.month
            self.df['quarter'] = self.df['timestamp'].dt.quarter
            self.df['year'] = self.df['timestamp'].dt.year

            # Cyclical encoding of time features
            self.df['hour_sin'] = np.sin(self.df['hour'] * (2 * np.pi / 24))
            self.df['hour_cos'] = np.cos(self.df['hour'] * (2 * np.pi / 24))
            self.df['day_of_week_sin'] = np.sin(self.df['day_of_week'] * (2 * np.pi / 7))
            self.df['day_of_week_cos'] = np.cos(self.df['day_of_week'] * (2 * np.pi / 7))
            self.df['month_sin'] = np.sin((self.df['month'] - 1) * (2 * np.pi / 12))
            self.df['month_cos'] = np.cos((self.df['month'] - 1) * (2 * np.pi / 12))
            self.df['day_sin'] = np.sin((self.df['day'] - 1) * (2 * np.pi / 31))
            self.df['day_cos'] = np.cos((self.df['day'] - 1) * (2 * np.pi / 31))

            # Time categorical features
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
            self.df['is_working_hours'] = ((self.df['hour'] >= 8) & (self.df['hour'] <= 18) & ~self.df['day_of_week'].isin([5, 6])).astype(int)
            self.df['is_night'] = ((self.df['hour'] < 6) | (self.df['hour'] >= 22)).astype(int)

            # Special day parts
            self.df['morning'] = ((self.df['hour'] >= 5) & (self.df['hour'] < 12)).astype(int)
            self.df['afternoon'] = ((self.df['hour'] >= 12) & (self.df['hour'] < 18)).astype(int)
            self.df['evening'] = ((self.df['hour'] >= 18) & (self.df['hour'] < 22)).astype(int)

            # Create lag features
            lags = [1, 2, 3, 6, 12, 24]
            for lag in lags:
                self.df[f'energy_lag_{lag}h'] = self.df['equipment_energy_consumption'].shift(lag)

            # Create rolling window features
            windows = [3, 6, 12, 24]
            for window in windows:
                self.df[f'energy_rolling_mean_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).mean()
                self.df[f'energy_rolling_std_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).std()
                self.df[f'energy_rolling_min_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).min()
                self.df[f'energy_rolling_max_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).max()
                self.df[f'energy_rolling_range_{window}h'] = self.df[f'energy_rolling_max_{window}h'] - self.df[f'energy_rolling_min_{window}h']

            # Create zone features
            temp_cols = [col for col in self.df.columns if 'temperature' in col and 'outdoor' not in col]
            if temp_cols:
                self.df['avg_zone_temp'] = self.df[temp_cols].mean(axis=1)
                self.df['min_zone_temp'] = self.df[temp_cols].min(axis=1)
                self.df['max_zone_temp'] = self.df[temp_cols].max(axis=1)
                self.df['zone_temp_range'] = self.df['max_zone_temp'] - self.df['min_zone_temp']

            humidity_cols = [col for col in self.df.columns if 'humidity' in col and 'outdoor' not in col]
            if humidity_cols:
                self.df['avg_zone_humidity'] = self.df[humidity_cols].mean(axis=1)
                self.df['min_zone_humidity'] = self.df[humidity_cols].min(axis=1)
                self.df['max_zone_humidity'] = self.df[humidity_cols].max(axis=1)
                self.df['zone_humidity_range'] = self.df['max_zone_humidity'] - self.df['min_zone_humidity']

            for i in range(1, 10):
                temp_col = f'zone{i}_temperature'
                hum_col = f'zone{i}_humidity'
                if temp_col in self.df.columns and hum_col in self.df.columns:
                    self.df[f'zone{i}_heat_index'] = self.df[temp_col] - 0.55 * (1 - self.df[hum_col]/100) * (self.df[temp_col] - 14.5)

            # Create weather features
            if 'outdoor_temperature' in self.df.columns and 'avg_zone_temp' in self.df.columns:
                self.df['indoor_outdoor_temp_diff'] = self.df['avg_zone_temp'] - self.df['outdoor_temperature']
                for i in range(1, 10):
                    temp_col = f'zone{i}_temperature'
                    if temp_col in self.df.columns:
                        self.df[f'zone{i}_outdoor_temp_diff'] = self.df[temp_col] - self.df['outdoor_temperature']

            if 'wind_speed' in self.df.columns and 'outdoor_temperature' in self.df.columns:
                self.df['wind_chill'] = self.df.apply(
                    lambda x: x['outdoor_temperature'] - (x['wind_speed'] * 0.7)
                    if x['outdoor_temperature'] < 10 and x['wind_speed'] > 0
                    else x['outdoor_temperature'],
                    axis=1
                )

            for col in ['avg_zone_temp', 'outdoor_temperature', 'atmospheric_pressure']:
                if col in self.df.columns:
                    self.df[f'{col}_change_1h'] = self.df[col] - self.df[col].shift(1)
                    self.df[f'{col}_rate_of_change'] = self.df[f'{col}_change_1h'] / 1.0

            # Drop random variables
            self.df = self.df.drop(['random_variable1', 'random_variable2'], axis=1, errors='ignore')

            # Drop rows with NaN from lag features
            self.df = self.df.dropna().reset_index(drop=True)

            logger.info("Feature engineering done.")
        except Exception as e:
            logger.error(f"Error while feature engineering: {e}")
            raise CustomException("Failed to engineer features", str(e))

    def handle_multicollinearity(self):
        try:
            # Prepare data for VIF calculation
            X = self.df.drop(columns=['equipment_energy_consumption', 'timestamp'], errors='ignore')
            if X.shape[0] <= 1 or X.shape[1] <= 1:
                raise ValueError("Insufficient data or features for VIF calculation")

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif_data = vif_data.sort_values("VIF", ascending=False)

            # Select top features (example: top 15)
            top_features = vif_data.head(15)["Feature"].tolist()
            self.selected_features = top_features
            logger.info(f"Selected features based on VIF: {self.selected_features}")

            # Reduce DataFrame to selected features plus target and timestamp
            self.df = self.df[self.selected_features + ['equipment_energy_consumption', 'timestamp']]
            logger.info("Multicollinearity handled.")
        except Exception as e:
            logger.error(f"Error while handling multicollinearity: {e}")
            raise CustomException("Failed to handle multicollinearity", str(e))

    def split_and_scale_data(self):
        try:
            # Sort by timestamp for time series split
            self.df = self.df.sort_values('timestamp')
            
            # Split data (80% train, 20% test)
            split_idx = int(len(self.df) * 0.8)
            train_df = self.df.iloc[:split_idx]
            test_df = self.df.iloc[split_idx:]

            X_train = train_df.drop(columns=['equipment_energy_consumption', 'timestamp'], errors='ignore')
            X_test = test_df.drop(columns=['equipment_energy_consumption', 'timestamp'], errors='ignore')
            y_train = train_df['equipment_energy_consumption']
            y_test = test_df['equipment_energy_consumption']

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            logger.info("Splitting and scaling done.")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logger.error(f"Error while splitting and scaling data: {e}")
            raise CustomException("Failed to split and scale data", str(e))

    def save_data_and_scaler(self, X_train, X_test, y_train, y_test):
        try:
            joblib.dump(X_train, os.path.join(self.output_path, 'X_train.pkl'))
            joblib.dump(X_test, os.path.join(self.output_path, 'X_test.pkl'))
            joblib.dump(y_train, os.path.join(self.output_path, 'y_train.pkl'))
            joblib.dump(y_test, os.path.join(self.output_path, 'y_test.pkl'))
            joblib.dump(self.scaler, os.path.join(self.output_path, 'scaler.pkl'))

            logger.info("All saving part is done...")
        except Exception as e:
            logger.error(f"Error while saving data: {e}")
            raise CustomException("Failed to save data", str(e))

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.feature_engineering()
        self.handle_multicollinearity()
        X_train_scaled, X_test_scaled, y_train, y_test = self.split_and_scale_data()
        self.save_data_and_scaler(X_train_scaled, X_test_scaled, y_train, y_test)

        logger.info("Data Preprocessing pipeline executed successfully...")

if __name__ == "__main__":
    input_path = r"C:\MLOps Projects\End-to-End-MLOps-Smart-Factory-Energy-Prediction\artifacts\raw\raw.csv"
    output_path = r"C:\MLOps Projects\End-to-End-MLOps-Smart-Factory-Energy-Prediction\artifacts\processed"

    processor = DataPreprocessing(input_path, output_path)
    processor.run()