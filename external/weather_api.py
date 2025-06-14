import pandas as pd
import numpy as np
import logging
from datetime import datetime
import yaml
import os
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mock_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock weather data for the given timestamps."""
    np.random.seed(42)  # For reproducibility
    
    # Extract unique dates from timestamps
    dates = pd.to_datetime(df['timestamp']).dt.date.unique()
    
    # Create a dictionary of weather conditions and their probabilities
    weather_conditions = {
        'Clear': 0.4,
        'Clouds': 0.3,
        'Rain': 0.2,
        'Thunderstorm': 0.05,
        'Snow': 0.05
    }
    
    # Generate daily weather data
    daily_weather = {}
    for date in dates:
        # Select weather condition based on probabilities
        condition = np.random.choice(
            list(weather_conditions.keys()),
            p=list(weather_conditions.values())
        )
        
        # Generate temperature based on condition
        if condition == 'Clear':
            temp = np.random.normal(28, 3)  # Warm, clear day
        elif condition == 'Clouds':
            temp = np.random.normal(25, 2)  # Mild, cloudy day
        elif condition == 'Rain':
            temp = np.random.normal(22, 2)  # Cool, rainy day
        elif condition == 'Thunderstorm':
            temp = np.random.normal(20, 2)  # Cool, stormy day
        else:  # Snow
            temp = np.random.normal(5, 2)   # Cold, snowy day
            
        daily_weather[date] = {
            'condition': condition,
            'temperature': round(temp, 1)
        }
    
    # Create weather DataFrame
    weather_data = []
    for idx, row in df.iterrows():
        date = pd.to_datetime(row['timestamp']).date()
        weather = daily_weather[date]
        weather_data.append({
            'timestamp': row['timestamp'],
            'weather_condition': weather['condition'],
            'temperature': weather['temperature']
        })
    
    return pd.DataFrame(weather_data)

def get_weather_data(df: pd.DataFrame, use_api: bool = False) -> Optional[pd.DataFrame]:
    """
    Get weather data either from API or mock data.
    
    Args:
        df: DataFrame containing timestamps
        use_api: Whether to use the actual API (default: False)
    
    Returns:
        DataFrame with weather data or None if API fails
    """
    if use_api:
        try:
            # Load API key from config
            config_path = os.path.join('config', 'api_keys.yaml')
            if not os.path.exists(config_path):
                raise ValueError("Please set your OpenWeatherMap API key in config/api_keys.yaml")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                api_key = config.get('openweathermap_api_key')
            
            if not api_key:
                raise ValueError("OpenWeatherMap API key not found in config")
            
            # TODO: Implement actual API call here
            # This is where the real API integration would go
            pass
            
        except Exception as e:
            logger.error(f"Error in weather data processing: {str(e)}")
            return None
    
    # Use mock data by default
    return get_mock_weather_data(df)

def add_weather_features(df: pd.DataFrame, use_api: bool = False) -> pd.DataFrame:
    """
    Add weather features to the DataFrame.
    
    Args:
        df: Input DataFrame
        use_api: Whether to use the actual API (default: False)
    
    Returns:
        DataFrame with added weather features
    """
    try:
        weather_df = get_weather_data(df, use_api)
        if weather_df is None:
            logger.warning("Using mock weather data due to API unavailability")
            weather_df = get_mock_weather_data(df)
        
        # Merge weather data with input DataFrame
        result_df = pd.merge(
            df,
            weather_df,
            on='timestamp',
            how='left'
        )
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error adding weather features: {str(e)}")
        # Fallback to mock data
        return pd.merge(
            df,
            get_mock_weather_data(df),
            on='timestamp',
            how='left'
        )

def get_weather_features(df: pd.DataFrame, use_api: bool = False) -> pd.DataFrame:
    """
    Get weather features for the given DataFrame.
    This is an alias for add_weather_features for backward compatibility.
    
    Args:
        df: Input DataFrame
        use_api: Whether to use the actual API (default: False)
    
    Returns:
        DataFrame with added weather features
    """
    return add_weather_features(df, use_api) 