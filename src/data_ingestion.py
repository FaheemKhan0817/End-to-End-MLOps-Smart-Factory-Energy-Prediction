import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Import paths from config
from config.paths_config import RAW_DIR, RAW_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, CONFIG_PATH

class DataIngestion:
    def __init__(self):
        # Load config.yaml to get the data source URL
        with open(CONFIG_PATH, 'r') as file:
            self.config = yaml.safe_load(file)
        self.data_source = self.config['data_ingestion']['data_source']

    def download_data(self):
        """Download the dataset from the GitHub URL and save it as raw.csv."""
        # Create raw directory if it doesn't exist
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # Download and save the dataset
        df = pd.read_csv(self.data_source)
        df.to_csv(RAW_FILE_PATH, index=False)
        print(f"Raw data saved at: {RAW_FILE_PATH}")
        return df

    def split_data(self, df, test_size=0.2, random_state=42):
        """Split the dataset into train and test sets (80:20 ratio)."""
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Save train and test datasets
        train_df.to_csv(TRAIN_FILE_PATH, index=False)
        test_df.to_csv(TEST_FILE_PATH, index=False)
        print(f"Train data saved at: {TRAIN_FILE_PATH}")
        print(f"Test data saved at: {TEST_FILE_PATH}")
        return train_df, test_df

    def ingest_data(self):
        """Main method to execute data ingestion and splitting."""
        # Download the data
        df = self.download_data()
        # Split the data
        train_df, test_df = self.split_data(df)
        return train_df, test_df

if __name__ == "__main__":
    # Instantiate and run the data ingestion process
    data_ingestion = DataIngestion()
    train_df, test_df = data_ingestion.ingest_data()