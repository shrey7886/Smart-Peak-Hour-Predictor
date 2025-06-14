import pandas as pd
import numpy as np
import logging
from typing import Union, Optional
from external.weather_api import add_weather_features
from external.holidays import get_holiday_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_and_preprocess(data: Union[str, pd.DataFrame], use_weather_api: bool = False) -> pd.DataFrame:
    """
    Validate and preprocess the input data.
    
    Args:
        data: Either a file path (str) or DataFrame containing the input data
        use_weather_api: Whether to use the actual weather API (default: False)
    
    Returns:
        Preprocessed DataFrame
    """
    try:
        logger.info(f"[DEBUG] Type of data argument: {type(data)}")
        if isinstance(data, pd.DataFrame):
            logger.info(f"[DEBUG] DataFrame columns: {data.columns}")
            logger.info(f"[DEBUG] DataFrame head:\n{data.head()}\n")
        elif isinstance(data, str):
            logger.info(f"[DEBUG] Data argument is a file path: {data}")
        else:
            logger.warning(f"[DEBUG] Data argument is neither DataFrame nor str: {data}")
        
        # Load data if file path is provided
        if isinstance(data, str):
            logger.info(f"Loading and validating data from {data}")
            df = pd.read_csv(data)
        else:
            logger.info("Using provided DataFrame")
            df = data.copy()
        
        # Validate required columns
        required_columns = ['timestamp', 'transactions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add basic time features
        logger.info("Adding basic time features")
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add weather features
        logger.info("Adding weather features")
        df = add_weather_features(df, use_api=use_weather_api)
        
        # Add holiday features
        logger.info("Adding holiday features")
        df = get_holiday_features(df)
        
        # Process categorical columns
        categorical_columns = ['weather_condition', 'holiday_type', 'holiday_name']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Regular')
                df[col] = df[col].astype('category')
        
        # Save processed data if input was a file path
        if isinstance(data, str):
            output_path = "data/processed_shop.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise
