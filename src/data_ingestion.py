import os
import pandas as pd
import yaml
import sys
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIR, RAW_FILE_PATH, CONFIG_PATH

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
        """Download the dataset from the specified source and save it as raw.csv."""
        try:
            logger.info("Starting data download from the specified source")
            
            # Create the raw directory if it doesn't exist
            os.makedirs(RAW_DIR, exist_ok=True)
            logger.info(f"Created raw data directory at: {RAW_DIR}")
            
            # Download the dataset and save it
            df = pd.read_csv(self.data_source)
            df.to_csv(RAW_FILE_PATH, index=False)
            logger.info(f"Successfully downloaded and saved raw data to: {RAW_FILE_PATH}")
            return df
        
        except Exception as e:
            logger.error("Error occurred while downloading data from the specified source")
            raise CustomException("Failed to download dataset", e)

    def ingest_data(self):
        """Run the full data ingestion process: download the data."""
        try:
            logger.info("Starting the data ingestion process")
            
            # Download the data
            df = self.download_data()
            
            logger.info("Data ingestion completed successfully")
            return df
        
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
        data_ingestion.ingest_data()
        logger.info("Data ingestion script completed")
    except CustomException as ce:
        logger.error(f"Script failed with CustomException: {str(ce)}")
    except Exception as e:
        logger.error(f"Script failed with unexpected error: {str(e)}")