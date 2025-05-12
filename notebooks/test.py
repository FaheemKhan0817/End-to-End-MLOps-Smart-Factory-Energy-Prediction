#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directories for outputs
for directory in ['models', 'results']:
    if not os.path.exists(directory):
        os.makedirs(directory)

print("Current Date and Time:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
print("Libraries imported and directories created!")

# Define file path for the dataset
input_csv_path = r"C:\ML Projects\DS-Intern-Assignment-Faheem-Khan\data\data.csv"

# Load the dataset
def load_data(file_path):
    """Load the dataset and convert data types."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert data types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded dataset with shape: {df.shape}")
    return df

# Load dataset
df = load_data(input_csv_path)

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check data types
print("\nData types after conversion:")
print(df.dtypes)

# Basic statistics of the dataset
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values)
print(f"Total missing values: {missing_values.sum()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Analyze the target variable
def analyze_target_variable(df):
    """Analyze the target variable distribution."""
    target = 'equipment_energy_consumption'
    
    print("\n=== Target Variable Analysis ===")
    print(f"Unique values in {target}: {df[target].nunique()}")
    print(f"Min: {df[target].min()}, Max: {df[target].max()}")
    print("Most common values:")
    print(df[target].value_counts().head(10))
    
    # Check for negative values
    negative_count = (df[target] < 0).sum()
    print(f"\nNegative energy consumption values: {negative_count} ({negative_count/len(df)*100:.2f}% of data)")
    
    # Plot histogram of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target], bins=30, kde=True)
    plt.title('Distribution of Equipment Energy Consumption')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.savefig('results/target_distribution.png')
    plt.show()

# Run target analysis
analyze_target_variable(df)

# Analyze correlations with target variable
def analyze_correlations(df):
    """Analyze correlations between features and target variable."""
    target = 'equipment_energy_consumption'
    correlations = df.corr()[target].sort_values(ascending=False)
    
    print("\n=== Feature Relationships with Target ===")
    print("Top 10 correlations with energy consumption:")
    print(correlations.head(11))  # 11 because it includes self-correlation
    
    print("\nBottom 5 correlations with energy consumption:")
    print(correlations.tail(5))
    
    # Check random variables specifically
    print("\n=== Random Variables Analysis ===")
    if 'random_variable1' in correlations:
        print(f"Correlation of random_variable1 with target: {correlations['random_variable1']:.4f}")
    if 'random_variable2' in correlations:
        print(f"Correlation of random_variable2 with target: {correlations['random_variable2']:.4f}")
    
    # Plot correlation heatmap for top features
    plt.figure(figsize=(14, 10))
    top_corr_features = correlations.head(15).index
    top_corr_df = df[top_corr_features].corr()
    sns.heatmap(top_corr_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap: Top Features vs Energy Consumption')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png')
    plt.show()
    
    return correlations

# Run correlation analysis
correlations = analyze_correlations(df)

# Fix negative energy consumption values
def fix_energy_consumption(df):
    """Fix negative energy consumption values."""
    print("\n=== Handling Negative Energy Consumption ===")
    
    # Get original statistics
    print("Statistics Before Correction:")
    print(df['equipment_energy_consumption'].describe())
    
    # Count negative values
    negative_count = (df['equipment_energy_consumption'] < 0).sum()
    print(f"Negative values: {negative_count} ({negative_count/len(df)*100:.2f}% of data)")
    
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Replace negative values with median of positive values
    positive_median = df[df['equipment_energy_consumption'] > 0]['equipment_energy_consumption'].median()
    df.loc[df['equipment_energy_consumption'] < 0, 'equipment_energy_consumption'] = positive_median
    
    # Get new statistics
    print("\nStatistics After Correction:")
    print(df['equipment_energy_consumption'].describe())
    
    # Plot before and after distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['equipment_energy_consumption'], bins=30, kde=True)
    plt.title('Energy Consumption After Correction')
    plt.savefig('results/energy_corrected.png')
    plt.show()
    
    return df

# Fix negative energy values
df = fix_energy_consumption(df)

# Handle missing values
def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n=== Handling Missing Values ===")
    
    df = df.copy()  # Create copy to avoid modifying original
    initial_missing = df.isnull().sum().sum()
    print(f"Initial missing values: {initial_missing}")
    
    # For target variable, drop rows with missing values
    df = df.dropna(subset=['equipment_energy_consumption'])
    
    # For sensor data, use time-based interpolation
    df_with_time_index = df.set_index('timestamp')
    
    # Sensor columns
    sensor_cols = [col for col in df_with_time_index.columns if any(x in col for x in ['temperature', 'humidity', 'pressure'])]
    for col in sensor_cols:
        df_with_time_index[col] = df_with_time_index[col].interpolate(method='time')
    
    # Reset index to get timestamp back as a column
    df = df_with_time_index.reset_index()
    
    # For remaining columns, use median imputation
    remaining_cols = [col for col in df.columns if col not in sensor_cols and col != 'timestamp' and col != 'equipment_energy_consumption']
    for col in remaining_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill any remaining NA values in target
    positive_median = df[df['equipment_energy_consumption'] > 0]['equipment_energy_consumption'].median()
    df['equipment_energy_consumption'] = df['equipment_energy_consumption'].fillna(positive_median)
    
    # drop missing values in zone3_humidity zone6_temperature   
    df = df.dropna(subset=['zone3_humidity', 'zone6_temperature'])
    
    
    # drop duplicate all raws
    df = df.drop_duplicates()
    
    final_missing = df.isnull().sum().sum()
    print(f"Remaining missing values: {final_missing}")
    
    return df

# Handle missing values
df = handle_missing_values(df)

# Fix implausible values
def fix_implausible_values(df):
    """Fix implausible values in the dataset."""
    print("\n=== Handling Implausible Values ===")
    
    df = df.copy()  # Create copy to avoid modifying original
    
    # Fix temperature values
    temp_cols = [col for col in df.columns if 'temperature' in col]
    for col in temp_cols:
        if 'outdoor' not in col:  # Indoor temperatures shouldn't be negative
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"Fixing {neg_count} negative values in {col}")
                df.loc[df[col] < 0, col] = df[df[col] > 0][col].median()
    
    # Fix humidity values (should be 0-100%)
    humidity_cols = [col for col in df.columns if 'humidity' in col]
    for col in humidity_cols:
        neg_count = (df[col] < 0).sum()
        high_count = (df[col] > 100).sum()
        if neg_count > 0 or high_count > 0:
            print(f"Fixing {neg_count} negative and {high_count} >100% values in {col}")
            df.loc[df[col] < 0, col] = df[df[col] >= 0 & (df[col] <= 100)][col].median()
            df.loc[df[col] > 100, col] = 100
    
    return df

# Fix implausible values
df = fix_implausible_values(df)

# Handle outliers
def handle_outliers(df):
    """Handle outliers in the dataset."""
    print("\n=== Handling Outliers ===")
    
    df = df.copy()  # Create copy to avoid modifying original
    
    # Don't modify the target variable
    cols_to_check = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_check = [col for col in cols_to_check if col != 'equipment_energy_consumption']
    
    # Use a conservative approach (5 std devs from mean)
    for col in cols_to_check:
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        lower_bound = mean_val - 5 * std_val
        upper_bound = mean_val + 5 * std_val
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Capping {outliers} extreme outliers in {col}")
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

# Handle outliers
df = handle_outliers(df)

# Final data quality check
print("\n=== Final Data Quality Check ===")
print(f"Shape after preprocessing: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Sort by timestamp for time series analysis
df = df.sort_values('timestamp').reset_index(drop=True)

# Display cleaned data
print("\nFirst 5 rows of cleaned data:")
print(df.head())

# Create time features
def create_time_features(df):
    """Create time-based features from timestamp."""
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
    df['day_sin'] = np.sin((df['day'] - 1) * (2 * np.pi / 31))
    df['day_cos'] = np.cos((df['day'] - 1) * (2 * np.pi / 31))
    
    # Time categorial features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_working_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & ~df['day_of_week'].isin([5, 6])).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    
    # Special day parts
    df['morning'] = ((df['hour'] >= 5) & (df['hour'] < 12)).astype(int)
    df['afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    
    return df

# Create time features
df = create_time_features(df)
print(f"Shape after adding time features: {df.shape}")

# Create lag features
def create_lag_features(df):
    """Create lag features for time series modeling."""
    df = df.copy()
    
    # Ensure dataframe is sorted by timestamp
    df = df.sort_values('timestamp')
    
    # Create lag features
    lags = [1, 2, 3, 6, 12, 24]  # 1, 2, 3, 6, 12, and 24 hours ago
    for lag in lags:
        df[f'energy_lag_{lag}h'] = df['equipment_energy_consumption'].shift(lag)
    
    # Create rolling window features
    windows = [3, 6, 12, 24]
    for window in windows:
        # Rolling mean (captures trend)
        df[f'energy_rolling_mean_{window}h'] = df['equipment_energy_consumption'].rolling(
            window=window, min_periods=1).mean()
        
        # Rolling std (captures volatility)
        df[f'energy_rolling_std_{window}h'] = df['equipment_energy_consumption'].rolling(
            window=window, min_periods=1).std()
        
        # Rolling min and max
        df[f'energy_rolling_min_{window}h'] = df['equipment_energy_consumption'].rolling(
            window=window, min_periods=1).min()
        df[f'energy_rolling_max_{window}h'] = df['equipment_energy_consumption'].rolling(
            window=window, min_periods=1).max()
        
        # Rolling range (max-min)
        df[f'energy_rolling_range_{window}h'] = (
            df[f'energy_rolling_max_{window}h'] - df[f'energy_rolling_min_{window}h']
        )
    
    return df

# Create lag features
df = create_lag_features(df)
print(f"Shape after adding lag features: {df.shape}")

# Create zone features
def create_zone_features(df):
    """Create features related to factory zones."""
    df = df.copy()
    
    # Zone temperature statistics
    temp_cols = [col for col in df.columns if 'temperature' in col and 'outdoor' not in col]
    if temp_cols:
        df['avg_zone_temp'] = df[temp_cols].mean(axis=1)
        df['min_zone_temp'] = df[temp_cols].min(axis=1)
        df['max_zone_temp'] = df[temp_cols].max(axis=1)
        df['zone_temp_range'] = df['max_zone_temp'] - df['min_zone_temp']
    
    # Zone humidity statistics
    humidity_cols = [col for col in df.columns if 'humidity' in col and 'outdoor' not in col]
    if humidity_cols:
        df['avg_zone_humidity'] = df[humidity_cols].mean(axis=1)
        df['min_zone_humidity'] = df[humidity_cols].min(axis=1)
        df['max_zone_humidity'] = df[humidity_cols].max(axis=1)
        df['zone_humidity_range'] = df['max_zone_humidity'] - df['min_zone_humidity']
    
    # Temperature-humidity interactions for each zone (heat index)
    for i in range(1, 10):
        temp_col = f'zone{i}_temperature'
        hum_col = f'zone{i}_humidity'
        
        if temp_col in df.columns and hum_col in df.columns:
            # Heat index approximation (temp-humidity interaction effect)
            df[f'zone{i}_heat_index'] = df[temp_col] - 0.55 * (1 - df[hum_col]/100) * (df[temp_col] - 14.5)
    
    return df

# Create zone features
df = create_zone_features(df)
print(f"Shape after adding zone features: {df.shape}")

# Create weather features
def create_weather_features(df):
    """Create weather-related features."""
    df = df.copy()
    
    # Indoor-outdoor temperature differential
    if 'outdoor_temperature' in df.columns and 'avg_zone_temp' in df.columns:
        df['indoor_outdoor_temp_diff'] = df['avg_zone_temp'] - df['outdoor_temperature']
        
        # Individual zone temperature differentials
        for i in range(1, 10):
            temp_col = f'zone{i}_temperature'
            if temp_col in df.columns:
                df[f'zone{i}_outdoor_temp_diff'] = df[temp_col] - df['outdoor_temperature']
    
    # Wind chill effect (approximation)
    if 'wind_speed' in df.columns and 'outdoor_temperature' in df.columns:
        df['wind_chill'] = df.apply(
            lambda x: x['outdoor_temperature'] - (x['wind_speed'] * 0.7) 
            if x['outdoor_temperature'] < 10 and x['wind_speed'] > 0 
            else x['outdoor_temperature'], 
            axis=1
        )
    
    # Differential features
    for col in ['avg_zone_temp', 'outdoor_temperature', 'atmospheric_pressure']:
        if col in df.columns:
            # Change from previous hour
            df[f'{col}_change_1h'] = df[col] - df[col].shift(1)
            
            # Rate of change (derivative)
            df[f'{col}_rate_of_change'] = df[f'{col}_change_1h'] / 1.0  # per hour
    
    return df

# Create weather features
df = create_weather_features(df)
print(f"Shape after adding weather features: {df.shape}")

# Remove random variables and finalize feature engineering
def finalize_features(df):
    """Finalize feature engineering and prepare for modeling."""
    df = df.copy()
    
    # Exclude random variables based on correlation analysis
    print("Excluding random variables due to very low correlation with target")
    df = df.drop(['random_variable1', 'random_variable2'], axis=1, errors='ignore')
    
    # Drop rows with NaN from lag features and reset index
    df = df.dropna().reset_index(drop=True)
    
    return df

# Finalize features
df = finalize_features(df)
print(f"Final dataset shape after feature engineering: {df.shape}")

# Calculate Variance Inflation Factor (VIF) for features
def calculate_vif(X):
    """Calculate VIF for each feature."""
    # Add constant
    X_with_const = pd.DataFrame({"const": np.ones(len(X))}, index=X.index)
    X_with_const = pd.concat([X_with_const, X], axis=1)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                        for i in range(X_with_const.shape[1])]
    
    return vif_data.sort_values("VIF", ascending=False)

# Prepare data for VIF calculation
X_for_vif = df.drop(['timestamp'], axis=1, errors='ignore')

# Calculate VIF
print("=== Multicollinearity Analysis with VIF ===")
vif_results = calculate_vif(X_for_vif)
print("\nVariance Inflation Factor (VIF) for each feature:")
print(vif_results)

# Save VIF results
vif_results.to_csv("results/vif_analysis.csv", index=False)

# Handle multicollinearity based on VIF analysis
def reduce_multicollinearity(df, vif_results, vif_threshold=100):
    """
    Reduce multicollinearity by selecting one feature from each highly collinear group.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    vif_results : pandas.DataFrame
        DataFrame with VIF scores for each feature
    vif_threshold : float
        Threshold for VIF scores to consider as high multicollinearity
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with reduced multicollinearity
    """
    print("\n=== Reducing Multicollinearity Based on VIF Analysis ===")
    
    # Create a copy of the dataframe
    df_reduced = df.copy()
    
    # Keep the target variable
    target = 'equipment_energy_consumption'
    
    # Create groups of collinear features based on VIF analysis
    # These groupings are based on the domain knowledge and feature names
    
    # Group 1: Zone temperature features (keep one per zone)
    zone_temp_features = [f'zone{i}_temperature' for i in range(1, 10)]
    
    # Group 2: Zone humidity features (keep one)
    zone_humidity_features = [f'zone{i}_humidity' for i in range(1, 10)]
    
    # Group 3: Zone heat index features (have lower VIF, keep them)
    zone_heat_index_features = [f'zone{i}_heat_index' for i in range(1, 10)]
    
    # Group 4: Lag features (keep only the most important ones)
    lag_features = ['energy_lag_1h', 'energy_lag_2h']  # Keep these two, drop others
    
    # Group 5: Rolling statistics (keep mean, drop others)
    rolling_mean_features = ['energy_rolling_mean_3h']  # Keep this one
    rolling_features_to_drop = [
        'energy_rolling_max_3h', 'energy_rolling_min_3h', 'energy_rolling_range_3h',
        'energy_rolling_max_6h', 'energy_rolling_min_6h', 'energy_rolling_range_6h',
        'energy_rolling_max_12h', 'energy_rolling_min_12h', 'energy_rolling_range_12h',
        'energy_rolling_max_24h', 'energy_rolling_min_24h', 'energy_rolling_range_24h'
    ]
    
    # Group 6: Temperature difference features (keep indoor_outdoor_temp_diff, drop zone-specific diffs)
    temp_diff_features_to_drop = [f'zone{i}_outdoor_temp_diff' for i in range(1, 10)]
    
    # Group 7: Rate of change features (all have high VIF, drop them)
    rate_change_features = [
        'atmospheric_pressure_rate_of_change', 'outdoor_temperature_rate_of_change',
        'avg_zone_temp_rate_of_change'
    ]
    
    # Group 8: Change features (keep simpler ones)
    change_features_to_drop = [
        'atmospheric_pressure_change_1h', 'outdoor_temperature_change_1h',
        'avg_zone_temp_change_1h'
    ]
    
    # Group 9: Zone statistics (some are duplicative)
    zone_stats_to_drop = [
        'max_zone_temp', 'min_zone_temp', 'zone_temp_range',
        'max_zone_humidity', 'min_zone_humidity', 'zone_humidity_range'
    ]
    
    # Combine all features to drop
    features_to_drop = (
        rolling_features_to_drop + 
        temp_diff_features_to_drop + 
        rate_change_features + 
        change_features_to_drop + 
        zone_stats_to_drop
    )
    
    # Features to keep (make sure they exist in the dataframe)
    features_to_keep = (
        zone_temp_features + 
        zone_heat_index_features + 
        ['avg_zone_temp', 'avg_zone_humidity', 'outdoor_temperature'] + 
        lag_features +
        rolling_mean_features +
        ['indoor_outdoor_temp_diff'] +
        ['hour_sin', 'hour_cos', 'is_working_hours']
    )
    
    # Check if features exist in the dataframe
    features_to_keep = [f for f in features_to_keep if f in df.columns]
    features_to_drop = [f for f in features_to_drop if f in df.columns]
    
    # Drop features with high multicollinearity
    df_reduced = df.drop(features_to_drop, axis=1, errors='ignore')
    
    # Make sure to keep timestamp and target
    if 'timestamp' not in features_to_keep and 'timestamp' in df.columns:
        features_to_keep.append('timestamp')
    if target not in features_to_keep and target in df.columns:
        features_to_keep.append(target)
    
    # For debugging, print the number of features kept
    print(f"Reduced features from {df.shape[1]} to {len(features_to_keep)} potential features")
    
    return df_reduced

# Use the function to reduce multicollinearity
df_reduced = reduce_multicollinearity(df, vif_results)

# Display remaining columns
print("\nRemaining columns after multicollinearity reduction:")
print(df_reduced.columns.tolist())

# Prepare data for modeling with reduced feature set
def prepare_model_data(df_reduced):
    """
    Prepare data for modeling by splitting into train and test sets.
    
    Parameters:
    -----------
    df_reduced : pandas.DataFrame
        Dataframe with reduced multicollinearity
        
    Returns:
    --------
    tuple
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    print("\n=== Preparing Data for Time Series Modeling ===")
    
    # Sort by timestamp to maintain time series integrity
    if 'timestamp' in df_reduced.columns:
        df_reduced = df_reduced.sort_values('timestamp')
    
    # Separate features and target
    X = df_reduced.drop(['timestamp', 'equipment_energy_consumption'], axis=1, errors='ignore')
    y = df_reduced['equipment_energy_consumption']
    
    # Split into training and testing sets (80% train, 20% test)
    split_idx = int(len(df_reduced) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features for models like ElasticNet
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Prepare data for modeling
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_model_data(df_reduced)

# Feature selection based on Lasso regularization
def select_important_features(X_train, X_test, X_train_scaled, X_test_scaled, y_train, max_features=20):
    """
    Select important features using Lasso regularization.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Test features
    X_train_scaled : pandas.DataFrame
        Scaled training features
    X_test_scaled : pandas.DataFrame
        Scaled test features
    y_train : pandas.Series
        Training target
    max_features : int, optional
        Maximum number of features to select, by default 20
        
    Returns:
    --------
    tuple
        X_train_selected, X_test_selected, X_train_scaled_selected, X_test_scaled_selected, selected_features
    """
    print(f"\n=== Selecting Top {max_features} Important Features ===")
    
    # Use Lasso for feature selection
    lasso = Lasso(alpha=0.01, random_state=42)
    selector = SelectFromModel(
        lasso,
        max_features=max_features,
        threshold=-np.inf  # Ensure exactly max_features are selected
    )
    
    # Fit the selector on scaled data (important for Lasso)
    selector.fit(X_train_scaled, y_train)
    
    # Get selected feature mask and names
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    
    # Create reduced feature datasets
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    X_train_scaled_selected = X_train_scaled[selected_features]
    X_test_scaled_selected = X_test_scaled[selected_features]
    
    print(f"Selected {len(selected_features)} features:")
    for feature in selected_features:
        print(f"- {feature}")
    
    return X_train_selected, X_test_selected, X_train_scaled_selected, X_test_scaled_selected, selected_features

# Select important features
X_train_selected, X_test_selected, X_train_scaled_selected, X_test_scaled_selected, selected_features = select_important_features(
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, max_features=15
)

# Train ElasticNet model on selected features
def train_elasticnet_model(X_train_scaled_selected, X_test_scaled_selected, y_train, y_test):
    """
    Train ElasticNet model on selected features.
    
    Parameters:
    -----------
    X_train_scaled_selected : pandas.DataFrame
        Selected scaled training features
    X_test_scaled_selected : pandas.DataFrame
        Selected scaled test features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    tuple
        model, y_pred, r2, rmse, mae
    """
    print("\n=== Training ElasticNet Model on Selected Features ===")
    
    # Create and train ElasticNet model
    model = ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=42
    )
    
    model.fit(X_train_scaled_selected, y_train)
    y_pred = model.predict(X_test_scaled_selected)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nElasticNet Model Performance:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return model, y_pred, r2, rmse, mae

# Train ElasticNet model
elasticnet_model, elasticnet_pred, elasticnet_r2, elasticnet_rmse, elasticnet_mae = train_elasticnet_model(
    X_train_scaled_selected, X_test_scaled_selected, y_train, y_test
)

# Train LightGBM model on selected features
def train_lightgbm_model(X_train_selected, X_test_selected, y_train, y_test):
    """
    Train LightGBM model on selected features.
    
    Parameters:
    -----------
    X_train_selected : pandas.DataFrame
        Selected training features
    X_test_selected : pandas.DataFrame
        Selected test features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    tuple
        model, y_pred, r2, rmse, mae
    """
    print("\n=== Training LightGBM Model on Selected Features ===")
    
    # Create and train LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nLightGBM Model Performance:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return model, y_pred, r2, rmse, mae

# Train LightGBM model
lightgbm_model, lightgbm_pred, lightgbm_r2, lightgbm_rmse, lightgbm_mae = train_lightgbm_model(
    X_train_selected, X_test_selected, y_train, y_test
)

# Train XGBoost model on selected features
def train_xgboost_model(X_train_selected, X_test_selected, y_train, y_test):
    """
    Train XGBoost model on selected features.
    
    Parameters:
    -----------
    X_train_selected : pandas.DataFrame
        Selected training features
    X_test_selected : pandas.DataFrame
        Selected test features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    tuple
        model, y_pred, r2, rmse, mae
    """
    print("\n=== Training XGBoost Model on Selected Features ===")
    
    # Create and train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nXGBoost Model Performance:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return model, y_pred, r2, rmse, mae

# Train XGBoost model
xgboost_model, xgboost_pred, xgboost_r2, xgboost_rmse, xgboost_mae = train_xgboost_model(
    X_train_selected, X_test_selected, y_train, y_test
)

# Compare models and select the best one
def compare_models(models_dict):
    """
    Compare models and select the best one.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model results
        
    Returns:
    --------
    tuple
        best_model_name, best_model, best_pred, best_r2
    """
    print("\n=== Model Comparison ===")
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Model': list(models_dict.keys()),
        'R²': [models_dict[m]['r2'] for m in models_dict.keys()],
        'RMSE': [models_dict[m]['rmse'] for m in models_dict.keys()],
        'MAE': [models_dict[m]['mae'] for m in models_dict.keys()]
    }).sort_values('R²', ascending=False).reset_index(drop=True)
    
    print(comparison)
    
    # Get best model
    best_model_name = comparison.iloc[0]['Model']
    best_model = models_dict[best_model_name]['model']
    best_pred = models_dict[best_model_name]['pred']
    best_r2 = models_dict[best_model_name]['r2']
    
    print(f"\nBest model: {best_model_name} with R² = {best_r2:.4f}")
    
    # Visualize model comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='R²', data=comparison)
    plt.title('Model Comparison: R² Score')
    plt.ylim(0.95, 1.0)  # Adjust as needed for your actual R² values
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='RMSE', data=comparison)
    plt.title('Model Comparison: RMSE')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.show()
    
    return best_model_name, best_model, best_pred, best_r2

# Collect model results
models_dict = {
    'ElasticNet': {
        'model': elasticnet_model,
        'pred': elasticnet_pred,
        'r2': elasticnet_r2,
        'rmse': elasticnet_rmse,
        'mae': elasticnet_mae
    },
    'LightGBM': {
        'model': lightgbm_model,
        'pred': lightgbm_pred,
        'r2': lightgbm_r2,
        'rmse': lightgbm_rmse,
        'mae': lightgbm_mae
    },
    'XGBoost': {
        'model': xgboost_model,
        'pred': xgboost_pred,
        'r2': xgboost_r2,
        'rmse': xgboost_rmse,
        'mae': xgboost_mae
    }
}

# Compare models and select the best one
best_model_name, best_model, best_pred, best_r2 = compare_models(models_dict)

# Analyze feature importance for the best model
def analyze_feature_importance(model, feature_names, model_name):
    """
    Analyze feature importance for the best model.
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
        
    Returns:
    --------
    pandas.DataFrame
        Feature importance dataframe
    """
    print(f"\n=== Feature Importance Analysis for {model_name} ===")
    
    # Get feature importance
    if model_name == 'ElasticNet':
        # For ElasticNet, use absolute coefficients as importance
        importance = np.abs(model.coef_)
        # Normalize to sum to 1 for comparison with tree-based models
        importance = importance / importance.sum() if importance.sum() > 0 else importance
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models like LightGBM and XGBoost
        importance = model.feature_importances_
    else:
        print(f"Model {model_name} doesn't provide feature importances")
        return None
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Display top features
    print("\nTop 15 Most Important Features:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"{i+1}. {row['Feature']} - {row['Importance']:.4f}")
    
    return feature_importance

# Get feature importance for the best model
feature_importance = analyze_feature_importance(best_model, X_train_selected.columns, best_model_name)

# Visualize predictions
def visualize_predictions(y_test, y_pred, model_name, n_samples=100):
    """
    Visualize actual vs predicted values.
    
    Parameters:
    -----------
    y_test : pandas.Series
        Test target
    y_pred : array
        Predicted values
    model_name : str
        Name of the model
    n_samples : int, optional
        Number of samples to plot, by default 100
    """
    print(f"\n=== Visualizing Predictions for {model_name} ===")
    
    # Plot actual vs predicted
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.values[:n_samples], label='Actual')
    plt.plot(y_pred[:n_samples], label='Predicted')
    plt.title(f'{model_name}: Actual vs Predicted Energy Consumption (First {n_samples} Test Points)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/predictions.png')
    plt.show()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('results/residuals.png')
    plt.show()
    
    # Calculate residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    print(f"Mean of residuals: {mean_residual:.4f}")
    print(f"Standard deviation of residuals: {std_residual:.4f}")

# Visualize predictions for the best model
visualize_predictions(y_test, best_pred, best_model_name)

# Generate insights based on feature importance
def generate_insights(feature_importance, model_name, r2):
    """
    Generate insights based on feature importance.
    
    Parameters:
    -----------
    feature_importance : pandas.DataFrame
        Feature importance dataframe
    model_name : str
        Name of the model
    r2 : float
        R² score of the model
    """
    print("\n=== Key Insights and Recommendations ===")
    print(f"Best model: {model_name} with R² = {r2:.4f}")
    
    if feature_importance is None:
        print("\nFeature importance data is not available.")
        return
    
    # Get top features
    top_features = feature_importance['Feature'].head(10).tolist()
    print("\nTop 10 factors affecting energy consumption:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}")
    
    # Check feature categories
    has_temp = any('temperature' in f for f in top_features)
    has_humidity = any('humidity' in f for f in top_features)
    has_heat_index = any('heat_index' in f for f in top_features)
    has_lag = any('lag' in f for f in top_features)
    has_rolling = any('rolling' in f for f in top_features)
    
    print("\nRecommendations for Energy Optimization:")
    
    if has_temp or has_heat_index:
        print("\n1. Temperature Management:")
        print("   - Optimize temperature setpoints in manufacturing zones")
        if has_heat_index:
            print("   - Pay attention to temperature-humidity interaction (heat index)")
    
    if has_lag or has_rolling:
        print("\n2. Time-Based Strategies:")
        print("   - Schedule energy-intensive operations during optimal times")
        print("   - Implement predictive control based on historical patterns")
    
    if has_humidity:
        print("\n3. Humidity Control:")
        print("   - Monitor and control humidity levels for optimal energy efficiency")
    
    print("\n4. Implementation Plan:")
    print("   - Create real-time monitoring dashboard using this model")
    print("   - Set up alert system for abnormal energy consumption patterns")
    print("   - Conduct regular model retraining with new data")

# Generate insights
generate_insights(feature_importance, best_model_name, best_r2)

# Save the best model
def save_model(model, model_name):
    """
    Save the model to a file.
    
    Parameters:
    -----------
    model : object
        Trained model
    model_name : str
        Name of the model
        
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create models directory if it doesn't exist
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model
    model_path = f'models/{model_name}_energy_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model_path

# Save the best model
model_path = save_model(best_model, best_model_name)

# Final message
print("\n=== Smart Factory Energy Prediction Project Complete ===")
print(f"Best Model: {best_model_name} with R² = {best_r2:.4f}")
print(f"Model saved to: {model_path}")
print("Analysis completed successfully!")

# Make predictions on new data
def load_model(model_path='models/ElasticNet_energy_model.pkl'):
    """
    Load the trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    object
        Loaded model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_test_data(n_samples=24):
    """
    Generate sample test data with the top 15 features.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    pandas.DataFrame
        Test data with top 15 features
    """
    # Initialize empty DataFrame
    test_data = pd.DataFrame()
    
    # Generate timestamps for time series (hourly data for one day)
    start_time = datetime(2025, 5, 8, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    test_data['timestamp'] = timestamps
    
    # Generate sample values for each feature
    
    # 1. Previous energy consumption - cyclic pattern
    base_energy = 100  # base energy value
    # Create a cyclical pattern similar to daily energy usage
    energy_cycle = np.sin(np.linspace(0, 2*np.pi, n_samples)) * 20 + base_energy
    energy_with_noise = energy_cycle + np.random.normal(0, 5, n_samples)
    
    # Fill lag features using the generated energy pattern
    test_data['energy_lag_1h'] = np.roll(energy_with_noise, 1)
    test_data['energy_lag_1h'][0] = energy_with_noise[0]  # Handle the first value
    
    test_data['energy_lag_2h'] = np.roll(energy_with_noise, 2)
    test_data['energy_lag_2h'][:2] = energy_with_noise[0]  # Handle the first two values
    
    # Calculate rolling mean for the 3-hour window
    test_data['energy_rolling_mean_3h'] = test_data['energy_lag_1h'].rolling(window=3, min_periods=1).mean()
    
    # 2. Zone temperatures (generally between 18-25°C)
    for i in [1, 2, 3, 4, 5, 6, 7, 9]:  # Only zones needed for top features
        base_temp = 21 + np.sin(np.linspace(0, 2*np.pi, n_samples)) * 2  # Daily cycle 19-23°C
        zone_temp = base_temp + np.random.normal(0, 0.5, n_samples)  # Add some noise
        test_data[f'zone{i}_temperature'] = zone_temp
    
    # 3. Zone humidity (only for zone6 and avg_zone_humidity)
    base_humidity = 50 + np.sin(np.linspace(0, 2*np.pi, n_samples)) * 5  # Daily cycle 45-55%
    zone_humidity = base_humidity + np.random.normal(0, 2, n_samples)  # Add some noise
    test_data['zone6_humidity'] = zone_humidity
    
    # Generate humidity for other zones to calculate avg_zone_humidity
    for i in range(1, 10):  # Still need all zones for avg_zone_humidity
        base_humidity = 50 + np.sin(np.linspace(0, 2*np.pi, n_samples)) * 5
        test_data[f'zone{i}_humidity'] = base_humidity + np.random.normal(0, 2, n_samples)
    
    # 4. Calculate zone heat indices (for zone1, zone3, zone9)
    for i in [1, 3, 9]:
        temp_col = f'zone{i}_temperature'
        hum_col = f'zone{i}_humidity'
        test_data[f'zone{i}_heat_index'] = test_data[temp_col] - 0.55 * (1 - test_data[hum_col]/100) * (test_data[temp_col] - 14.5)
    
    # 5. Calculate average humidity
    test_data['avg_zone_humidity'] = test_data[[f'zone{i}_humidity' for i in range(1, 10)]].mean(axis=1)
    
    # Select only the top 15 features for prediction
    test_features = test_data[top_features]
    
    return test_data, test_features

def predict_energy(model, test_features, scaler=None):
    """
    Make energy consumption predictions using the trained model.
    
    Parameters:
    -----------
    model : object
        Trained model
    test_features : pandas.DataFrame
        Test features
    scaler : object, optional
        Scaler for feature normalization, by default None
        
    Returns:
    --------
    numpy.ndarray
        Predicted energy consumption
    """
    # Scale features if scaler is provided (required for ElasticNet)
    if scaler:
        test_features_scaled = scaler.transform(test_features)
        predictions = model.predict(test_features_scaled)
    else:
        predictions = model.predict(test_features)
    
    return predictions

def visualize_predictions(test_data, predictions):
    """
    Visualize the predictions.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with timestamp
    predictions : numpy.ndarray
        Predicted energy consumption
    """
    plt.figure(figsize=(15, 6))
    
    # Plot predicted energy consumption
    plt.plot(test_data['timestamp'], predictions, marker='o', linestyle='-', color='blue', label='Predicted Energy')
    
    # Format the plot
    plt.title('Predicted Energy Consumption', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Energy Consumption', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    
    # Save the plot
    plt.savefig('results/test_predictions.png')
    plt.show()
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Mean predicted energy consumption: {np.mean(predictions):.2f}")
    print(f"Min predicted energy consumption: {np.min(predictions):.2f}")
    print(f"Max predicted energy consumption: {np.max(predictions):.2f}")

# Main execution
# Main execution
if __name__ == "__main__":
    print("=== Smart Factory Energy Prediction - Test Model ===")
    print(f"Current Date and Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User: FaheemKhan0817")
    
    # Load the model
    model = load_model('models/ElasticNet_energy_model.pkl')
    
    if model:
        # Create a simple scaler for demonstration (normally you would load the saved scaler)
        scaler = StandardScaler()
        
        # Generate test data
        test_data, test_features = generate_test_data(n_samples=24)
        
        print("\nTest features:")
        print(test_features.head())
        
        # Fit scaler on test data (normally would use the same scaler used for training)
        scaler.fit(test_features)
        
        # Make predictions
        predictions = predict_energy(model, test_features, scaler)
        
        print("\nPredictions:")
        print(predictions[:10])
        
        # Visualize predictions
        visualize_predictions(test_data, predictions)
        
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'timestamp': test_data['timestamp'],
            'predicted_energy': predictions
        })
        results_df.to_csv('results/test_predictions.csv', index=False)
        print("\nPredictions saved to results/test_predictions.csv")
    
    print("\n=== Test Complete ===")


# In[ ]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# Ensure the directory for saving models exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset (assuming df is your cleaned DataFrame)
# df = pd.read_csv('path_to_your_cleaned_data.csv')

# Top features selected from feature importance analysis
top_features = [
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

# Prepare the data using only the top features
X = df[top_features]
y = df['equipment_energy_consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the ElasticNet model
elasticnet_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
elasticnet_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = elasticnet_model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"ElasticNet Model Performance with Top Features:")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Save the trained model
model_path = 'models/ElasticNet_energy_model_top_features.pkl'
joblib.dump(elasticnet_model, model_path)
print(f"Model saved to {model_path}")

# Save the scaler for future use
scaler_path = 'models/ElasticNet_scaler_top_features.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

