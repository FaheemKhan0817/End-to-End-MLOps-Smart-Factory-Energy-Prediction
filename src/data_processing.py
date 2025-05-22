import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_FILE_PATH, PROCESSED_DIR, SCALER_PATH
import joblib

class DataPreprocessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.selected_features = None
        self.top_features = [
            'energy_rolling_mean_3h', 'energy_lag_2h', 'energy_lag_1h', 'zone3_heat_index',
            'zone3_temperature', 'avg_zone_humidity', 'zone1_heat_index', 'zone1_temperature',
            'zone2_temperature', 'zone6_temperature'
        ]
        os.makedirs(output_path, exist_ok=True)
        self.logger = get_logger(__name__)
        self.logger.info("Data Preprocessing Initialized...")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            for col in self.df.select_dtypes(include=['object']).columns:
                if col != 'timestamp':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    self.logger.warning(f"Non-numeric values in {col} converted to NaN")
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            self.logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", str(e))

    def fix_energy_consumption(self):
        try:
            target = 'equipment_energy_consumption'
            negative_count = (self.df[target] < 0).sum()
            if negative_count > 0:
                self.logger.info(f"Fixing {negative_count} negative values in {target}")
                positive_median = self.df[self.df[target] > 0][target].median()
                self.df.loc[self.df[target] < 0, target] = positive_median
        except Exception as e:
            self.logger.error(f"Error while fixing energy consumption: {e}")
            raise CustomException("Failed to fix energy consumption", str(e))

    def handle_missing_values(self):
        try:
            initial_missing = self.df.isnull().sum().sum()
            self.logger.info(f"Initial missing values: {initial_missing}")
            self.df = self.df.dropna(subset=['equipment_energy_consumption'])
            self.df = self.df.set_index('timestamp')
            sensor_cols = [col for col in self.df.columns if any(x in col for x in ['temperature', 'humidity', 'pressure'])]
            for col in sensor_cols:
                self.df[col] = self.df[col].interpolate(method='time')
            self.df = self.df.reset_index()
            remaining_cols = [col for col in self.df.columns if col not in sensor_cols and col != 'timestamp' and col != 'equipment_energy_consumption']
            for col in remaining_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            self.df = self.df.dropna(subset=['zone3_humidity', 'zone6_temperature'])
            self.df = self.df.drop_duplicates()
            final_missing = self.df.isnull().sum().sum()
            self.logger.info(f"Remaining missing values: {final_missing}")
        except Exception as e:
            self.logger.error(f"Error while handling missing values: {e}")
            raise CustomException("Failed to handle missing values", str(e))

    def fix_implausible_values(self):
        try:
            temp_cols = [col for col in self.df.columns if 'temperature' in col and 'outdoor' not in col]
            for col in temp_cols:
                neg_count = (self.df[col] < 0).sum()
                if neg_count > 0:
                    self.logger.info(f"Fixing {neg_count} negative values in {col}")
                    self.df.loc[self.df[col] < 0, col] = self.df[self.df[col] > 0][col].median()
            humidity_cols = [col for col in self.df.columns if 'humidity' in col]
            for col in humidity_cols:
                neg_count = (self.df[col] < 0).sum()
                high_count = (self.df[col] > 100).sum()
                if neg_count > 0 or high_count > 0:
                    self.logger.info(f"Fixing {neg_count} negative and {high_count} >100% values in {col}")
                    median_valid = self.df[(self.df[col] >= 0) & (self.df[col] <= 100)][col].median()
                    self.df.loc[self.df[col] < 0, col] = median_valid
                    self.df.loc[self.df[col] > 100, col] = 100
        except Exception as e:
            self.logger.error(f"Error while fixing implausible values: {e}")
            raise CustomException("Failed to fix implausible values", str(e))

    def handle_outliers(self):
        try:
            cols_to_check = self.df.select_dtypes(include=['float64', 'int64']).columns
            cols_to_check = [col for col in cols_to_check if col != 'equipment_energy_consumption']
            for col in cols_to_check:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    self.logger.info(f"Capping {outliers} outliers in {col}")
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        except Exception as e:
            self.logger.error(f"Error while handling outliers: {e}")
            raise CustomException("Failed to handle outliers", str(e))

    def create_time_features(self):
        try:
            new_columns = {}
            new_columns['hour'] = self.df['timestamp'].dt.hour
            new_columns['day'] = self.df['timestamp'].dt.day
            new_columns['day_of_week'] = self.df['timestamp'].dt.dayofweek
            new_columns['month'] = self.df['timestamp'].dt.month
            new_columns['quarter'] = self.df['timestamp'].dt.quarter
            new_columns['year'] = self.df['timestamp'].dt.year
            new_columns['hour_sin'] = np.sin(new_columns['hour'] * (2 * np.pi / 24))
            new_columns['hour_cos'] = np.cos(new_columns['hour'] * (2 * np.pi / 24))
            new_columns['day_of_week_sin'] = np.sin(new_columns['day_of_week'] * (2 * np.pi / 7))
            new_columns['day_of_week_cos'] = np.cos(new_columns['day_of_week'] * (2 * np.pi / 7))
            new_columns['month_sin'] = np.sin((new_columns['month'] - 1) * (2 * np.pi / 12))
            new_columns['month_cos'] = np.cos((new_columns['month'] - 1) * (2 * np.pi / 12))
            new_columns['day_sin'] = np.sin((new_columns['day'] - 1) * (2 * np.pi / 31))
            new_columns['day_cos'] = np.cos((new_columns['day'] - 1) * (2 * np.pi / 31))
            new_columns['is_weekend'] = new_columns['day_of_week'].isin([5, 6]).astype(int)
            new_columns['is_working_hours'] = ((new_columns['hour'] >= 8) & (new_columns['hour'] <= 18) & ~new_columns['day_of_week'].isin([5, 6])).astype(int)
            new_columns['is_night'] = ((new_columns['hour'] < 6) | (new_columns['hour'] >= 22)).astype(int)
            new_columns['morning'] = ((new_columns['hour'] >= 5) & (new_columns['hour'] < 12)).astype(int)
            new_columns['afternoon'] = ((new_columns['hour'] >= 12) & (new_columns['hour'] < 18)).astype(int)
            new_columns['evening'] = ((new_columns['hour'] >= 18) & (new_columns['hour'] < 22)).astype(int)
            self.df = pd.concat([self.df, pd.DataFrame(new_columns, index=self.df.index)], axis=1)
            self.logger.info("Time features created.")
        except Exception as e:
            self.logger.error(f"Error while creating time features: {e}")
            raise CustomException("Failed to create time features", str(e))

    def create_lag_features(self):
        try:
            lags = [1, 2, 3, 6, 12, 24]
            windows = [3, 6, 12, 24]
            new_columns = {}
            for lag in lags:
                new_columns[f'energy_lag_{lag}h'] = self.df['equipment_energy_consumption'].shift(lag)
            for window in windows:
                new_columns[f'energy_rolling_mean_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).mean()
                new_columns[f'energy_rolling_std_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).std()
                new_columns[f'energy_rolling_min_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).min()
                new_columns[f'energy_rolling_max_{window}h'] = self.df['equipment_energy_consumption'].rolling(window=window, min_periods=1).max()
                new_columns[f'energy_rolling_range_{window}h'] = new_columns[f'energy_rolling_max_{window}h'] - new_columns[f'energy_rolling_min_{window}h']
            self.df = pd.concat([self.df, pd.DataFrame(new_columns, index=self.df.index)], axis=1)
            self.logger.info("Lag features created.")
        except Exception as e:
            self.logger.error(f"Error while creating lag features: {e}")
            raise CustomException("Failed to create lag features", str(e))

    def create_zone_features(self):
        try:
            new_columns = {}
            temp_cols = [col for col in self.df.columns if 'temperature' in col and 'outdoor' not in col]
            if temp_cols:
                new_columns['avg_zone_temp'] = self.df[temp_cols].mean(axis=1)
                new_columns['min_zone_temp'] = self.df[temp_cols].min(axis=1)
                new_columns['max_zone_temp'] = self.df[temp_cols].max(axis=1)
                new_columns['zone_temp_range'] = new_columns['max_zone_temp'] - new_columns['min_zone_temp']
            humidity_cols = [col for col in self.df.columns if 'humidity' in col and 'outdoor' not in col]
            if humidity_cols:
                new_columns['avg_zone_humidity'] = self.df[humidity_cols].mean(axis=1)
                new_columns['min_zone_humidity'] = self.df[humidity_cols].min(axis=1)
                new_columns['max_zone_humidity'] = self.df[humidity_cols].max(axis=1)
                new_columns['zone_humidity_range'] = new_columns['max_zone_humidity'] - new_columns['min_zone_humidity']
            for i in range(1, 10):
                temp_col = f'zone{i}_temperature'
                hum_col = f'zone{i}_humidity'
                if temp_col in self.df.columns and hum_col in self.df.columns:
                    new_columns[f'zone{i}_heat_index'] = self.df[temp_col] - 0.55 * (1 - self.df[hum_col]/100) * (self.df[temp_col] - 14.5)
            self.df = pd.concat([self.df, pd.DataFrame(new_columns, index=self.df.index)], axis=1)
            self.logger.info("Zone features created.")
        except Exception as e:
            self.logger.error(f"Error while creating zone features: {e}")
            raise CustomException("Failed to create zone features", str(e))

    def create_weather_features(self):
        try:
            new_columns = {}
            if 'outdoor_temperature' in self.df.columns and 'avg_zone_temp' in self.df.columns:
                new_columns['indoor_outdoor_temp_diff'] = self.df['avg_zone_temp'] - self.df['outdoor_temperature']
                for i in range(1, 10):
                    temp_col = f'zone{i}_temperature'
                    if temp_col in self.df.columns:
                        new_columns[f'zone{i}_outdoor_temp_diff'] = self.df[temp_col] - self.df['outdoor_temperature']
            if 'wind_speed' in self.df.columns and 'outdoor_temperature' in self.df.columns:
                new_columns['wind_chill'] = self.df.apply(
                    lambda x: x['outdoor_temperature'] - (x['wind_speed'] * 0.7) if x['outdoor_temperature'] < 10 and x['wind_speed'] > 0 else x['outdoor_temperature'], axis=1)
            for col in ['avg_zone_temp', 'outdoor_temperature', 'atmospheric_pressure']:
                if col in self.df.columns:
                    new_columns[f'{col}_change_1h'] = self.df[col] - self.df[col].shift(1)
                    new_columns[f'{col}_rate_of_change'] = new_columns[f'{col}_change_1h'] / 1.0
            self.df = pd.concat([self.df, pd.DataFrame(new_columns, index=self.df.index)], axis=1)
            self.logger.info("Weather features created.")
        except Exception as e:
            self.logger.error(f"Error while creating weather features: {e}")
            raise CustomException("Failed to create weather features", str(e))

    def finalize_features(self):
        try:
            self.df = self.df.drop(['random_variable1', 'random_variable2'], axis=1, errors='ignore')
            self.df = self.df.dropna().reset_index(drop=True)
            self.logger.info("Feature engineering done.")
        except Exception as e:
            self.logger.error(f"Error while finalizing features: {e}")
            raise CustomException("Failed to finalize features", str(e))

    def reduce_multicollinearity(self):
        try:
            features_to_drop = [
                'energy_rolling_max_3h', 'energy_rolling_min_3h', 'energy_rolling_range_3h',
                'energy_rolling_max_6h', 'energy_rolling_min_6h', 'energy_rolling_range_6h',
                'energy_rolling_max_12h', 'energy_rolling_min_12h', 'energy_rolling_range_12h',
                'energy_rolling_max_24h', 'energy_rolling_min_24h', 'energy_rolling_range_24h',
                'energy_rolling_std_3h', 'energy_rolling_std_6h', 'energy_rolling_std_12h', 'energy_rolling_std_24h',
                'energy_lag_3h', 'energy_lag_6h', 'energy_lag_12h', 'energy_lag_24h',
                'zone1_outdoor_temp_diff', 'zone2_outdoor_temp_diff', 'zone3_outdoor_temp_diff',
                'zone4_outdoor_temp_diff', 'zone5_outdoor_temp_diff', 'zone6_outdoor_temp_diff',
                'zone7_outdoor_temp_diff', 'zone8_outdoor_temp_diff', 'zone9_outdoor_temp_diff',
                'atmospheric_pressure_rate_of_change', 'outdoor_temperature_rate_of_change',
                'avg_zone_temp_rate_of_change', 'atmospheric_pressure_change_1h',
                'outdoor_temperature_change_1h', 'avg_zone_temp_change_1h',
                'max_zone_temp', 'min_zone_temp', 'zone_temp_range',
                'max_zone_humidity', 'min_zone_humidity', 'zone_humidity_range'
            ]
            self.df = self.df.drop([f for f in features_to_drop if f in self.df.columns], axis=1, errors='ignore')
            self.logger.info(f"Reduced features to mitigate multicollinearity. Shape: {self.df.shape}")
        except Exception as e:
            self.logger.error(f"Error while reducing multicollinearity: {e}")
            raise CustomException("Failed to reduce multicollinearity", str(e))

    def select_important_features(self, max_features=15):
        try:
            X = self.df.drop(['timestamp', 'equipment_energy_consumption'], axis=1, errors='ignore')
            y = self.df['equipment_energy_consumption']
            split_idx = int(len(self.df) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
            selector = SelectFromModel(lasso, max_features=max_features, threshold=-np.inf)
            selector.fit(X_train, y_train)
            self.selected_features = X_train.columns[selector.get_support()].tolist()
            for feature in self.top_features:
                if feature in X_train.columns and feature not in self.selected_features:
                    self.selected_features.append(feature)
                    self.logger.info(f"Added critical feature: {feature}")
            self.df = self.df[self.selected_features + ['timestamp', 'equipment_energy_consumption']]
            self.logger.info(f"Selected features: {self.selected_features}")
        except Exception as e:
            self.logger.error(f"Error while selecting important features: {e}")
            raise CustomException("Failed to select important features", str(e))

    def prepare_model_data(self):
        try:
            self.df = self.df.sort_values('timestamp')
            split_idx = int(len(self.df) * 0.8)
            train_df = self.df.iloc[:split_idx]
            test_df = self.df.iloc[split_idx:]
            X_train = train_df[self.top_features]
            X_test = test_df[self.top_features]
            y_train = train_df['equipment_energy_consumption']
            y_test = test_df['equipment_energy_consumption']
            missing_features = [f for f in self.top_features if f not in X_train.columns]
            if missing_features:
                self.logger.error(f"Missing required features: {missing_features}")
                raise CustomException("Required features missing", f"Missing: {missing_features}")
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            joblib.dump(scaler, SCALER_PATH)
            self.logger.info(f"Scaler fitted on top features and saved to {SCALER_PATH}")
            self.logger.info(f"Top 10 features used for training: {self.top_features}")
            self.logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error while preparing model data: {e}")
            raise CustomException("Failed to prepare model data", str(e))

    def run(self):
        try:
            self.load_data()
            self.logger.info(f"Initial dataset shape: {self.df.shape}")
            self.fix_energy_consumption()
            self.handle_missing_values()
            self.fix_implausible_values()
            self.handle_outliers()
            self.create_time_features()
            self.create_lag_features()
            self.create_zone_features()
            self.create_weather_features()
            self.finalize_features()
            self.reduce_multicollinearity()
            self.select_important_features()
            X_train, X_test, y_train, y_test = self.prepare_model_data()
            self.logger.info(f"Final dataset shape: {self.df.shape}")
            self.logger.info("Data Preprocessing pipeline executed successfully...")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error in data preprocessing pipeline: {e}")
            raise CustomException("Failed to execute data preprocessing pipeline", str(e))

if __name__ == "__main__":
    preprocessor = DataPreprocessing(RAW_FILE_PATH, PROCESSED_DIR)
    X_train, X_test, y_train, y_test = preprocessor.run()