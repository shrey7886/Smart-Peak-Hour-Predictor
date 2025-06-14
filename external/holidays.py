import holidays
import pandas as pd
from typing import Union, List, Dict, Tuple
import logging
from datetime import datetime, timedelta, date

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_upcoming_holidays(country: str = 'IN', days_ahead: int = 30) -> List[Tuple[date, str, str]]:
    """
    Get a list of upcoming holidays for the specified country.
    
    Args:
        country (str): Country code for holidays (default: 'IN' for India)
        days_ahead (int): Number of days to look ahead for holidays
        
    Returns:
        List of tuples containing (date, holiday_name, holiday_type)
    """
    today = date.today()
    end_date = today + timedelta(days=days_ahead)
    
    # Initialize holidays for the date range
    country_holidays = holidays.CountryHoliday(country, years=range(today.year, end_date.year + 1))
    
    # Get all holidays in the range
    upcoming = []
    current_date = today
    while current_date <= end_date:
        if current_date in country_holidays:
            holiday_name = country_holidays[current_date]
            # Determine holiday type
            if current_date.weekday() >= 5:  # Weekend
                holiday_type = "Weekend"
            else:
                holiday_type = "Holiday"
            upcoming.append((current_date, holiday_name, holiday_type))
        current_date += timedelta(days=1)
    
    return upcoming

def add_holiday_flags(df: pd.DataFrame, country: str = 'IN') -> pd.DataFrame:
    """
    Add holiday flags to the dataframe based on the specified country's calendar.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'timestamp' column
        country (str): Country code for holidays (default: 'IN' for India)
        
    Returns:
        pd.DataFrame: Original dataframe with added holiday features
    """
    try:
        logger.info(f"Adding holiday flags for country: {country}")
        
        # Get the date range from the dataframe
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        # Initialize holidays for the date range
        country_holidays = holidays.CountryHoliday(country, years=range(min_date.year, max_date.year + 1))
        
        # Add holiday flags
        df["is_holiday"] = df["timestamp"].dt.date.apply(lambda x: x in country_holidays).astype(int)
        
        # Add holiday names where applicable
        df["holiday_name"] = df["timestamp"].dt.date.apply(
            lambda x: country_holidays.get(x, "Not a holiday")
        )
        
        # Add weekend flag
        df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday
        
        # Add special events (customizable based on business needs)
        special_events = {
            "New Year's Eve": (12, 31),
            "Valentine's Day": (2, 14),
            "Black Friday": (11, 24),  # Example date, adjust as needed
            "Christmas Eve": (12, 24),
            "Diwali": None,  # Will be handled by the holidays library for India
            "Holi": None,    # Will be handled by the holidays library for India
            "Eid": None      # Will be handled by the holidays library for India
        }
        
        # Add special event flags
        df["is_special_event"] = 0
        for event_name, event_date in special_events.items():
            if event_date is None:
                # These are handled by the holidays library
                continue
            month, day = event_date
            mask = (df["timestamp"].dt.month == month) & (df["timestamp"].dt.day == day)
            df.loc[mask, "is_special_event"] = 1
            df.loc[mask, "holiday_name"] = event_name
        
        # Combine all flags to determine holiday type
        df["holiday_type"] = df.apply(
            lambda row: "Special Event" if row["is_special_event"] == 1 else
                       ("Weekend" if row["is_weekend"] == 1 else 
                       ("Holiday" if row["is_holiday"] == 1 else "Regular")),
            axis=1
        )
        
        # Add days until next holiday
        df["days_until_holiday"] = df["timestamp"].dt.date.apply(
            lambda x: min((h[0] - x).days for h in get_upcoming_holidays(country) if h[0] > x)
            if any(h[0] > x for h in get_upcoming_holidays(country)) else 0
        )
        
        logger.info(f"Successfully added holiday features for {df['is_holiday'].sum()} holidays")
        return df
        
    except Exception as e:
        logger.error(f"Error adding holiday features: {str(e)}")
        raise

def get_holiday_features(df: pd.DataFrame, country: str = 'IN') -> pd.DataFrame:
    """
    Convenience wrapper around add_holiday_flags.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'timestamp' column
        country (str): Country code for holidays (default: 'IN' for India)
        
    Returns:
        pd.DataFrame: Original dataframe with added holiday features
    """
    try:
        return add_holiday_flags(df, country)
    except Exception as e:
        logger.error(f"Error in get_holiday_features: {str(e)}")
        raise 