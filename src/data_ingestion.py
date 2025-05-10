import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIR, RAW_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, CONFIG_PATH

# Set up the logger for this module
logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        """Initialize the DataIngestion class by loading the configuration."""
        try:
            logger.info("Loading configuration for data ingestion")
            with open(CONFIG_PATH, 'r') as file:
                self.config = yaml.safe_load(file)
            self.data_source = self.config['data_ingestion']['data_source']
            logger.info(f"Data source URL loaded: {self.data_source}")
        except Exception as e:
            logger.error("Failed to load configuration from config.yaml")
            raise CustomException("Could not load configuration file", e)

    def download_data(self):
        """Download the dataset from GitHub and save it as raw.csv."""
        try:
            logger.info("Starting data download from GitHub")
            
            # Create the raw directory if it doesn't exist
            os.makedirs(RAW_DIR, exist_ok=True)
            logger.info(f"Created raw data directory at: {RAW_DIR}")
            
            # Download the dataset and save it
            df = pd.read_csv(self.data_source)
            df.to_csv(RAW_FILE_PATH, index=False)
            logger.info(f"Successfully downloaded and saved raw data to: {RAW_FILE_PATH}")
            return df
        
        except Exception as e:
            logger.error("Error occurred while downloading data from GitHub")
            raise CustomException("Failed to download dataset", e)

    def split_data(self, df):
        """Split the dataset into train and test sets (80:20) and save them."""
        try:
            logger.info("Starting the train-test split process")
            
            # Split the data with 80:20 ratio
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test datasets
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info(f"Train data saved to: {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to: {TEST_FILE_PATH}")
            return train_df, test_df
        
        except Exception as e:
            logger.error("Error occurred while splitting the dataset")
            raise CustomException("Failed to split dataset into train and test sets", e)

    def ingest_data(self):
        """Run the full data ingestion process: download and split."""
        try:
            logger.info("Starting the data ingestion process")
            
            # Download and split the data
            df = self.download_data()
            train_df, test_df = self.split_data(df)
            
            logger.info("Data ingestion completed successfully")
            return train_df, test_df
        
        except CustomException as ce:
            logger.error(f"CustomException occurred: {str(ce)}")
            raise
        
        except Exception as e:
            logger.error("Unexpected error during data ingestion")
            raise CustomException("Data ingestion process failed", e)

if __name__ == "__main__":
    try:
        logger.info("Initiating data ingestion script")
        data_ingestion = DataIngestion()
        train_df, test_df = data_ingestion.ingest_data()
        logger.info("Data ingestion script completed")
    except CustomException as ce:
        logger.error(f"Script failed with CustomException: {str(ce)}")
    except Exception as e:
        logger.error(f"Script failed with unexpected error: {str(e)}")