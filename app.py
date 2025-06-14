# Import required modules first
import streamlit as st
import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime, timedelta
import io
import yaml
import plotly.graph_objects as go
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config - must be the first Streamlit command
st.set_page_config(page_title="📈 Smart Peak Hour Predictor", layout="wide")

# Import application modules
from utils.preprocessing import validate_and_preprocess
from external.weather_api import get_weather_features  # Changed from add_weather_features
from external.holidays import get_holiday_features
from utils.visualize import (
    display_weather_strip,
    display_holiday_info,
    create_weather_impact_chart,
    create_holiday_impact_chart,
    display_weather_holiday_dashboard,
    display_shop_dashboard_v2,
    display_forecast,
    display_weather_info,
    display_peak_hours,
    display_transaction_history
)
from predict_tft import predict_peak_hours
from utils.db import (
    init_db, 
    save_shop_profile, 
    get_shop_profile, 
    save_forecast, 
    get_forecast_history, 
    get_all_shops, 
    create_shop_directories,
    DatabaseManager
)
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from models.model_manager import ModelManager
from models.transfer_learning import TransferLearningManager
from models.online_learning import OnlineLearningManager

# Initialize session state
if 'current_shop' not in st.session_state:
    st.session_state.current_shop = None
if 'shop_data' not in st.session_state:
    st.session_state.shop_data = None

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def display_shop_selector():
    """Display shop selection interface with history"""
    st.sidebar.title("🏪 Shop Management")
    
    # Add a search box for shops
    search_query = st.sidebar.text_input("🔍 Search Shops", "")
    
    # Get all existing shops
    all_shops = get_all_shops()
    
    # Filter shops based on search query
    if search_query:
        all_shops = [shop for shop in all_shops if search_query.lower() in shop['shop_id'].lower()]
    
    # Display shop statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Shop Statistics")
    st.sidebar.metric("Total Shops", len(all_shops))
    active_shops = len([shop for shop in all_shops if 'last_upload' in shop and shop['last_upload']])
    st.sidebar.metric("Active Shops", active_shops)
    
    if all_shops:
        st.sidebar.markdown("### 📋 Existing Shops")
        for shop in all_shops:
            with st.sidebar.expander(f"🏪 {shop['shop_id']}", expanded=False):
                # Shop details in a more organized way
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Created:**")
                    st.markdown(f"*{pd.to_datetime(shop['created_at']).strftime('%Y-%m-%d')}*")
                with col2:
                    st.markdown("**Last Update:**")
                    if 'last_upload' in shop and shop['last_upload']:
                        st.markdown(f"*{pd.to_datetime(shop['last_upload']).strftime('%Y-%m-%d %H:%M')}*")
                    else:
                        st.markdown("*Never*")
                
                # Quick actions
                if st.button("📊 View Dashboard", key=f"view_{shop['shop_id']}"):
                    st.session_state.current_shop = shop['shop_id']
                    initialize_managers(shop['shop_id'])
                    st.rerun()
                
                if st.button("📈 View History", key=f"history_{shop['shop_id']}"):
                    st.session_state.current_shop = shop['shop_id']
                    st.session_state.show_history = True
                    initialize_managers(shop['shop_id'])
                    st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ➕ Add New Shop")
    
    with st.sidebar.form("new_shop_form"):
        new_shop_id = st.text_input(
            "Enter Shop ID",
            help="Enter a unique identifier for your shop (e.g., 'bangalore-shop-1')"
        )
        location = st.text_input(
            "Location",
            help="Enter the shop's location (e.g., 'Bangalore, Karnataka')"
        )
        shop_type = st.selectbox(
            "Shop Type",
            ["Retail", "Restaurant", "Cafe", "Supermarket", "Other"],
            help="Select the type of business"
        )
        
        submitted = st.form_submit_button("Create New Shop")
        if submitted and new_shop_id:
            if any(shop['shop_id'] == new_shop_id for shop in all_shops):
                st.sidebar.error("❌ Shop ID already exists! Please choose a different ID.")
            else:
                # Create new shop profile with additional details
                shop = {
                    'shop_id': new_shop_id,
                    'name': new_shop_id,  # Use shop_id as name initially
                    'location': location,
                    'type': shop_type,
                    'created_at': datetime.now().isoformat(),
                    'settings': {
                        'timezone': 'Asia/Kolkata',
                        'business_hours': {
                            'start': '09:00',
                            'end': '21:00'
                        }
                    }
                }
                save_shop_profile(shop['shop_id'], shop['name'], shop['location'])
                st.session_state.current_shop = new_shop_id
                initialize_managers(new_shop_id)
                st.sidebar.success(f"✅ Successfully created shop: {new_shop_id}")
                st.rerun()

def initialize_managers(shop_id):
    """Initialize all managers for a shop"""
    st.session_state.model_manager = ModelManager(shop_id)
    st.session_state.transfer_manager = TransferLearningManager()
    st.session_state.online_manager = OnlineLearningManager()

def display_predictions(predictions, df):
    """Display predictions and analytics in the dashboard."""
    try:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🌦️ Weather Impact", "📊 Analytics"])
        
        with tab1:
            # Display predictions chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions["timestamp"],
                y=predictions["transactions"],
                mode="lines",
                name="Predicted Transactions",
                line=dict(color="blue")
            ))
            
            # Add peak hours markers
            peak_hours = predictions[predictions["is_peak"] == 1]
            fig.add_trace(go.Scatter(
                x=peak_hours["timestamp"],
                y=peak_hours["transactions"],
                mode="markers",
                name="Peak Hours",
                marker=dict(
                    color="red",
                    size=10,
                    symbol="star"
                )
            ))
            
            fig.update_layout(
                title="Transaction Predictions with Peak Hours",
                xaxis_title="Time",
                yaxis_title="Predicted Transactions",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display weather strip below the chart
            display_weather_strip(predictions)
        
        with tab2:
            # Display weather impact analysis
            create_weather_impact_chart(predictions)
            create_holiday_impact_chart(predictions)
        
        with tab3:
            # Display comprehensive analytics dashboard
            display_weather_holiday_dashboard(predictions)
            
    except Exception as e:
        logger.error(f"Error displaying predictions: {str(e)}")
        st.error(f"An error occurred while displaying predictions: {str(e)}")

def main():
    """Main application entry point."""
    try:
        # Initialize database
        db = DatabaseManager()
        db.init_db()

        # Display shop selector in sidebar
        display_shop_selector()

        # Get current shop from session state
        current_shop = st.session_state.get('current_shop')
        if not current_shop:
            st.info("👈 Please select a shop from the sidebar to view its dashboard")
            return

        # Get shop profile
        shop_profile = get_shop_profile(current_shop)
        if not shop_profile:
            st.error(f"Shop {current_shop} not found")
            return

        # Get shop data
        df = None
        if st.session_state.get('shop_data') is not None:
            df = st.session_state.shop_data
        else:
            # Try to load from processed data
            try:
                df = pd.read_csv(f"data/shops/{current_shop}/processed_data.csv")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.session_state.shop_data = df
            except Exception as e:
                logger.error(f"Error loading shop data: {str(e)}")
                st.warning("No data available for this shop. Please upload data first.")
                return

        # Preprocess data if needed
        if df is not None and not df.empty:
            try:
                df = validate_and_preprocess(df)
                st.session_state.shop_data = df
            except Exception as e:
                logger.error(f"Error in preprocessing: {str(e)}")
                st.error("Error processing data. Using raw data instead.")

        # Check if model exists
        model_path = "models/shop_tft.ckpt"
        if not os.path.exists(model_path):
            st.warning("⚠️ Model not found. Please train the model first.")
            # Display dashboard without predictions
            display_shop_dashboard_v2(
                df=df,
                shop_id=current_shop,
                shop_name=shop_profile.get('name', current_shop),
                db=db,
                skip_predictions=True
            )
        else:
            # Display dashboard with predictions
            display_shop_dashboard_v2(
                df=df,
                shop_id=current_shop,
                shop_name=shop_profile.get('name', current_shop),
                db=db
            )

    except Exception as e:
        logger.error(f"Error in main app: {str(e)}")
        st.error("An error occurred. Please check the logs.")

if __name__ == "__main__":
    main()

# ───────────── Streamlit UI Setup ─────────────
st.title("📊 Smart Peak Hour Predictor")

# ───────────── API Key Setup ─────────────
st.sidebar.header("⚙️ Configuration")

def load_api_key():
    try:
        with open("config/api_keys.yaml", "r") as file:
            API_KEYS = yaml.safe_load(file)
        return API_KEYS.get("openweathermap")
    except Exception as e:
        return None

def get_mock_weather_data(df):
    """Generate mock weather data when no API key is available"""
    np.random.seed(42)  # For reproducibility
    weather_types = ['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm']
    mock_data = pd.DataFrame(index=df.index)
    
    # Generate realistic weather patterns
    mock_data['weather_main'] = np.random.choice(weather_types, size=len(df), p=[0.4, 0.3, 0.2, 0.05, 0.05])
    mock_data['temp'] = np.random.normal(25, 5, size=len(df))  # Mean 25°C, std 5°C
    mock_data['humidity'] = np.random.uniform(40, 90, size=len(df))
    mock_data['rain'] = np.where(mock_data['weather_main'] == 'Rain', 
                                np.random.uniform(0.1, 5, size=len(df)), 0)
    mock_data['snow'] = np.where(mock_data['weather_main'] == 'Snow', 
                                np.random.uniform(0.1, 2, size=len(df)), 0)
    mock_data['wind_speed'] = np.random.uniform(0, 20, size=len(df))
    mock_data['clouds'] = np.random.uniform(0, 100, size=len(df))
    
    return mock_data

def get_mock_holiday_data(df):
    """Generate mock holiday data when no API key is available"""
    np.random.seed(42)  # For reproducibility
    
    # Get the date range from the dataframe
    start_date = df['timestamp'].min().date()
    end_date = df['timestamp'].max().date()
    
    # Create a list of common holidays
    common_holidays = {
        'New Year\'s Day': (1, 1),
        'Valentine\'s Day': (2, 14),
        'International Women\'s Day': (3, 8),
        'Earth Day': (4, 22),
        'Labor Day': (5, 1),
        'Independence Day': (7, 4),
        'Halloween': (10, 31),
        'Christmas Eve': (12, 24),
        'Christmas Day': (12, 25),
        'New Year\'s Eve': (12, 31)
    }
    
    # Create a DataFrame with the same index as input
    mock_data = pd.DataFrame(index=df.index)
    
    # Initialize holiday columns
    mock_data['holiday_type'] = 'Regular'
    mock_data['is_holiday'] = 0
    mock_data['holiday_name'] = 'Not a holiday'
    
    # Add weekends using the timestamp column
    weekend_mask = df['timestamp'].dt.dayofweek.isin([5, 6])
    mock_data.loc[weekend_mask, 'holiday_type'] = 'Weekend'
    
    # Add holidays
    for holiday_name, (month, day) in common_holidays.items():
        holiday_mask = (df['timestamp'].dt.month == month) & (df['timestamp'].dt.day == day)
        mock_data.loc[holiday_mask, 'holiday_type'] = 'Holiday'
        mock_data.loc[holiday_mask, 'is_holiday'] = 1
        mock_data.loc[holiday_mask, 'holiday_name'] = holiday_name
    
    # Add some special events (randomly distributed)
    special_event_days = np.random.choice(len(df), size=min(5, len(df)), replace=False)
    mock_data.iloc[special_event_days, mock_data.columns.get_loc('holiday_type')] = 'Special Event'
    mock_data.iloc[special_event_days, mock_data.columns.get_loc('is_holiday')] = 1
    mock_data.iloc[special_event_days, mock_data.columns.get_loc('holiday_name')] = 'Special Event'
    
    return mock_data

api_key = load_api_key()
if not api_key or api_key == "YOUR_API_KEY_HERE":
    st.sidebar.info("ℹ️ Using mock weather and holiday data. For real-time data, add your OpenWeatherMap API key to config/api_keys.yaml")
    st.sidebar.markdown("""
    To enable real-time weather data:
    1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
    2. Get your API key
    3. Add it to `config/api_keys.yaml`:
    ```yaml
    openweathermap: "YOUR_API_KEY_HERE"
    ```
    """)
    use_mock_data = True
else:
    use_mock_data = False

# ───────────── Location Setup ─────────────
st.sidebar.subheader("📍 Location Settings")

# Define location mappings
LOCATION_MAPPINGS = {
    "IN": {  # India
        # States
        "Andhra Pradesh": {"lat": 17.38, "lon": 78.48, "city": "Hyderabad"},
        "Arunachal Pradesh": {"lat": 27.08, "lon": 93.60, "city": "Itanagar"},
        "Assam": {"lat": 26.14, "lon": 91.79, "city": "Guwahati"},
        "Bihar": {"lat": 25.59, "lon": 85.13, "city": "Patna"},
        "Chhattisgarh": {"lat": 21.27, "lon": 81.86, "city": "Raipur"},
        "Goa": {"lat": 15.49, "lon": 73.82, "city": "Panaji"},
        "Gujarat": {"lat": 23.03, "lon": 72.58, "city": "Ahmedabad"},
        "Haryana": {"lat": 28.61, "lon": 77.20, "city": "Gurgaon"},
        "Himachal Pradesh": {"lat": 31.10, "lon": 77.17, "city": "Shimla"},
        "Jharkhand": {"lat": 23.35, "lon": 85.33, "city": "Ranchi"},
        "Karnataka": {"lat": 12.97, "lon": 77.59, "city": "Bangalore"},
        "Kerala": {"lat": 10.52, "lon": 76.21, "city": "Kochi"},
        "Madhya Pradesh": {"lat": 23.25, "lon": 77.41, "city": "Bhopal"},
        "Maharashtra": {"lat": 19.07, "lon": 72.87, "city": "Mumbai"},
        "Manipur": {"lat": 24.82, "lon": 93.94, "city": "Imphal"},
        "Meghalaya": {"lat": 25.57, "lon": 91.88, "city": "Shillong"},
        "Mizoram": {"lat": 23.73, "lon": 92.71, "city": "Aizawl"},
        "Nagaland": {"lat": 25.67, "lon": 94.11, "city": "Kohima"},
        "Odisha": {"lat": 20.29, "lon": 85.82, "city": "Bhubaneswar"},
        "Punjab": {"lat": 30.73, "lon": 76.77, "city": "Chandigarh"},
        "Rajasthan": {"lat": 26.91, "lon": 75.78, "city": "Jaipur"},
        "Sikkim": {"lat": 27.33, "lon": 88.61, "city": "Gangtok"},
        "Tamil Nadu": {"lat": 13.08, "lon": 80.27, "city": "Chennai"},
        "Telangana": {"lat": 17.38, "lon": 78.48, "city": "Hyderabad"},
        "Tripura": {"lat": 23.84, "lon": 91.27, "city": "Agartala"},
        "Uttar Pradesh": {"lat": 26.85, "lon": 80.94, "city": "Lucknow"},
        "Uttarakhand": {"lat": 30.32, "lon": 78.03, "city": "Dehradun"},
        "West Bengal": {"lat": 22.57, "lon": 88.36, "city": "Kolkata"},
        
        # Union Territories
        "Andaman and Nicobar Islands": {"lat": 11.67, "lon": 92.74, "city": "Port Blair"},
        "Chandigarh": {"lat": 30.73, "lon": 76.77, "city": "Chandigarh"},
        "Dadra and Nagar Haveli and Daman and Diu": {"lat": 20.42, "lon": 72.83, "city": "Daman"},
        "Delhi": {"lat": 28.61, "lon": 77.20, "city": "New Delhi"},
        "Jammu and Kashmir": {"lat": 34.08, "lon": 74.80, "city": "Srinagar"},
        "Ladakh": {"lat": 34.15, "lon": 77.57, "city": "Leh"},
        "Lakshadweep": {"lat": 10.56, "lon": 72.63, "city": "Kavaratti"},
        "Puducherry": {"lat": 11.94, "lon": 79.83, "city": "Puducherry"}
    },
    "US": {  # United States
        "California": {"lat": 37.77, "lon": -122.41, "city": "San Francisco"},
        "New York": {"lat": 40.71, "lon": -74.00, "city": "New York City"},
        "Texas": {"lat": 29.76, "lon": -95.36, "city": "Houston"},
        "Florida": {"lat": 25.76, "lon": -80.19, "city": "Miami"},
        "Illinois": {"lat": 41.87, "lon": -87.62, "city": "Chicago"}
    },
    "GB": {  # United Kingdom
        "England": {"lat": 51.50, "lon": -0.11, "city": "London"},
        "Scotland": {"lat": 55.95, "lon": -3.18, "city": "Edinburgh"},
        "Wales": {"lat": 51.48, "lon": -3.17, "city": "Cardiff"},
        "Northern Ireland": {"lat": 54.59, "lon": -5.93, "city": "Belfast"}
    },
    "CA": {  # Canada
        "Ontario": {"lat": 43.65, "lon": -79.38, "city": "Toronto"},
        "Quebec": {"lat": 45.50, "lon": -73.56, "city": "Montreal"},
        "British Columbia": {"lat": 49.28, "lon": -123.12, "city": "Vancouver"},
        "Alberta": {"lat": 51.04, "lon": -114.07, "city": "Calgary"}
    },
    "AU": {  # Australia
        "New South Wales": {"lat": -33.86, "lon": 151.20, "city": "Sydney"},
        "Victoria": {"lat": -37.81, "lon": 144.96, "city": "Melbourne"},
        "Queensland": {"lat": -27.46, "lon": 153.02, "city": "Brisbane"},
        "Western Australia": {"lat": -31.95, "lon": 115.86, "city": "Perth"}
    }
}

# Country selection with full names
country_names = {
    "IN": "India",
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia"
}

# Simple two-dropdown interface
country = st.sidebar.selectbox(
    "Country",
    options=list(LOCATION_MAPPINGS.keys()),
    format_func=lambda x: country_names[x],
    help="Select your country"
)

state = st.sidebar.selectbox(
    "State/Region",
    options=list(LOCATION_MAPPINGS[country].keys()),
    help=f"Select your state/region in {country_names[country]}"
)

# Get coordinates for selected location
location_data = LOCATION_MAPPINGS[country][state]
lat = location_data["lat"]
lon = location_data["lon"]
city = location_data["city"]

# Display selected location info
st.sidebar.info(f"📍 Selected Location: {city}, {state}, {country_names[country]}")

REQUIRED_COLUMNS = [
    'timestamp', 'transactions', 'promotion_flag', 'promotion_type',
    'staff_count', 'event_flag', 'event_name', 'inventory_alert'
]

# ───────────── Step 1: Upload CSV ─────────────
st.header("📥 Step 1: Upload Sales CSV")

def generate_template_csv():
    # Create sample data for the template
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
    template_data = {
        'timestamp': dates,
        'transactions': np.random.randint(10, 50, size=24),  # Sample transaction counts
        'promotion_flag': np.random.choice([0, 1], size=24),  # Binary flag
        'promotion_type': np.random.choice(['None', 'Discount', 'Bundle', 'Flash Sale'], size=24),
        'staff_count': np.random.randint(2, 6, size=24),  # Sample staff counts
        'event_flag': np.random.choice([0, 1], size=24),  # Binary flag
        'event_name': np.random.choice(['None', 'Holiday', 'Special Sale', 'Seasonal Event'], size=24),
        'inventory_alert': np.random.choice([0, 1], size=24)  # Binary flag
    }
    df = pd.DataFrame(template_data)
    return df

# Add download template button
if st.button("📋 Download Template CSV"):
    template_df = generate_template_csv()
    csv = template_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Click to download template",
        data=csv,
        file_name="sales_template.csv",
        mime="text/csv",
        help="Download a template CSV file with the required columns and sample data"
    )

uploaded_file = st.file_uploader("Upload your hourly sales CSV file", type="csv")

if uploaded_file:
    file_path = os.path.join("data", "uploaded_shop.csv")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded successfully!")

    df_check = pd.read_csv(file_path, nrows=1)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_check.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
    else:
        if st.button("🔄 Preprocess Data"):
            try:
                # Load and preprocess the data
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add basic time features
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                
                # Create proper integer time index
                df = df.sort_values('timestamp')  # Sort by timestamp first
                df['time_idx'] = np.arange(len(df))  # Create sequential integer index
                
                # Add weather and holiday features
                if use_mock_data:
                    weather_data = get_mock_weather_data(df)
                    holiday_data = get_mock_holiday_data(df)
                    df = pd.concat([df, weather_data, holiday_data], axis=1)
                else:
                    # Use real API data with selected location
                    df = validate_and_preprocess(file_path, lat=lat, lon=lon, country=country)
                    # Ensure time_idx is integer after preprocessing
                    df['time_idx'] = np.arange(len(df))
                
                # Save processed data
                df.to_csv("data/processed_shop.csv", index=False)
                
                # Show data insights with location context
                st.success(f"✅ Data preprocessed successfully for {city}, {state}!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
                with col3:
                    st.metric("Average Daily Transactions", f"{df['transactions'].mean():.1f}")
                with col4:
                    st.metric("Weather Data Points", f"{df['temp'].notna().sum()}")

                # Show sample data with weather and holiday info
                st.subheader("📋 Sample Data")
                display_cols = ['timestamp', 'transactions', 'weather_main', 'temp', 'holiday_type', 'is_holiday']
                st.dataframe(df[display_cols].head())

                # Show data distribution
                st.subheader("📊 Data Distribution")
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Transactions distribution
                sns.histplot(data=df, x='transactions', bins=30, ax=ax1)
                ax1.set_title('Transaction Distribution')
                
                # Transactions by hour
                hourly_avg = df.groupby('hour')['transactions'].mean()
                sns.barplot(x=hourly_avg.index, y=hourly_avg.values, ax=ax2)
                ax2.set_title('Average Transactions by Hour')
                ax2.set_xlabel('Hour of Day')
                ax2.set_ylabel('Average Transactions')

                # Weather impact
                sns.boxplot(data=df, x='weather_main', y='transactions', ax=ax3)
                ax3.set_title('Transactions by Weather Condition')
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

                # Holiday impact
                sns.boxplot(data=df, x='holiday_type', y='transactions', ax=ax4)
                ax4.set_title('Transactions by Day Type')
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Preprocessing error: {e}")

        # ───────────── Step 2: Forecasting ─────────────
        if st.button("🔮 Predict Peak Hours"):
            st.subheader("📈 Prediction Results")

            try:
                # Load and validate data
                df = pd.read_csv("data/processed_shop.csv")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure time_idx is integer
                df['time_idx'] = df['time_idx'].astype(int)
                
                # If using mock data and weather/holiday features are missing, add them
                if use_mock_data and 'weather_main' not in df.columns:
                    weather_data = get_mock_weather_data(df)
                    holiday_data = get_mock_holiday_data(df)
                    df = pd.concat([df, weather_data, holiday_data], axis=1)
                
                # Add shop identifier and handle categorical columns
                df["shop"] = "shop_1"
                df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
                df["event_name"] = df["event_name"].astype(str).fillna("None")
                df["weather_main"] = df["weather_main"].astype(str).fillna("Clear")

                # Ensure all required columns are present and properly formatted
                required_reals = [
                    "time_idx", "transactions", "hour", "day_of_week", "is_weekend",
                    "staff_count", "promotion_flag", "event_flag", "inventory_alert",
                    "temp", "humidity", "rain", "snow", "wind_speed", "clouds",
                    "is_holiday"
                ]
                
                # Verify and convert all required real columns to float
                for col in required_reals:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with missing values and sort
                df = df.dropna(subset=required_reals)
                df = df.sort_values("time_idx")

                # Verify time_idx is sequential
                if not (df['time_idx'].diff().dropna() == 1).all():
                    df['time_idx'] = np.arange(len(df))

                # TimeSeriesDataSet parameters
                max_encoder_length = 24
                max_prediction_length = 6
                top_n_peaks = 3

                training_cutoff = df["time_idx"].max() - max_prediction_length
                
                # Create training dataset
                training = TimeSeriesDataSet(
                    df[df["time_idx"] <= training_cutoff],
                    time_idx="time_idx",
                    target="transactions",
                    group_ids=["shop"],
                    time_varying_known_reals=[
                        "time_idx", "hour", "day_of_week", "is_weekend",
                        "staff_count", "promotion_flag", "event_flag", "inventory_alert",
                        # Weather features
                        "temp", "humidity", "rain", "snow", "wind_speed", "clouds",
                        # Holiday features
                        "is_holiday"
                    ],
                    time_varying_unknown_reals=["transactions"],
                    time_varying_known_categoricals=[
                        "promotion_type", "event_name", "weather_main", "holiday_type"
                    ],
                    categorical_encoders={
                        "promotion_type": NaNLabelEncoder(add_nan=True),
                        "event_name": NaNLabelEncoder(add_nan=True),
                        "weather_main": NaNLabelEncoder(add_nan=True),
                        "holiday_type": NaNLabelEncoder(add_nan=True)
                    },
                    max_encoder_length=max_encoder_length,
                    max_prediction_length=max_prediction_length,
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                )

                # Create prediction dataset
                prediction = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
                prediction_dl = prediction.to_dataloader(train=False, batch_size=1, num_workers=0)

                # Load model
                model_path = "models/shop_tft.ckpt"
                if not os.path.exists(model_path):
                    st.error("❌ Model checkpoint not found. Please train it using Phase 2.")
                    st.stop()

                model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=torch.device("cpu"))

                # Make predictions
                predictions = model.predict(prediction_dl)
                predictions = predictions.cpu().numpy()

                # Get the last time index and create future time steps
                last_time_idx = df['time_idx'].max()
                future_time_steps = range(last_time_idx + 1, last_time_idx + max_prediction_length + 1)

                # Create results dataframe with actual timestamps
                last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
                future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(max_prediction_length)]

                result_df = pd.DataFrame({
                        "Timestamp": future_timestamps,
                        "Hour": [ts.hour for ts in future_timestamps],
                    "Predicted_Transactions": predictions[-max_prediction_length:].flatten(),
                    "Weather": df["weather_main"].iloc[-max_prediction_length:].values,
                    "Temperature": df["temp"].iloc[-max_prediction_length:].values,
                    "Holiday_Type": df["holiday_type"].iloc[-max_prediction_length:].values,
                    "Rain": df["rain"].iloc[-max_prediction_length:].values
                    })

                # Identify peak hours
                top_indices = result_df["Predicted_Transactions"].argsort()[-top_n_peaks:][::-1]
                peak_hours = result_df.iloc[top_indices].copy()

                # Add suggestions with weather and holiday context
                def get_suggestion(row):
                    base_suggestion = "📈 Add staff/stock!" if row["Predicted_Transactions"] > result_df["Predicted_Transactions"].quantile(0.75) else "✅ Normal"
                    weather_context = f" ({row['Weather']}, {row['Temperature']:.1f}°C)"
                    holiday_context = f" - {row['Holiday_Type']}"
                    return base_suggestion + weather_context + holiday_context

                result_df["Suggestion"] = result_df.apply(get_suggestion, axis=1)

                # Calculate confidence levels
                mean_pred = result_df["Predicted_Transactions"].mean()
                std_pred = result_df["Predicted_Transactions"].std()
                result_df["Confidence"] = result_df["Predicted_Transactions"].apply(
                    lambda x: "High" if x > mean_pred + std_pred else 
                    ("Medium" if x > mean_pred else "Low")
                )

                # Update peak_hours with confidence levels
                peak_hours["Confidence"] = result_df.iloc[top_indices]["Confidence"].values
                peak_hours["Suggestion"] = result_df.iloc[top_indices]["Suggestion"].values

                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Peak Transaction Hour", 
                            f"{peak_hours.iloc[0]['Hour']}:00",
                            f"{peak_hours.iloc[0]['Predicted_Transactions']:.1f} transactions")
                with col2:
                    avg_pred = result_df["Predicted_Transactions"].mean()
                    st.metric("Average Predicted Transactions", 
                            f"{avg_pred:.1f}",
                            f"{(avg_pred - df['transactions'].mean()):.1f} vs historical")
                with col3:
                    busy_hours = len(result_df[result_df["Suggestion"].str.startswith("📈")])
                    st.metric("Busy Hours Ahead", 
                            f"{busy_hours}",
                            f"{(busy_hours/len(result_df))*100:.0f}% of predicted period")
                with col4:
                    st.metric("Weather Impact", 
                            f"{result_df['Weather'].mode().iloc[0]}",
                            f"Avg Temp: {result_df['Temperature'].mean():.1f}°C")

                # Display detailed predictions
                st.subheader("🕒 Hourly Predictions")
                display_cols = ["Timestamp", "Hour", "Predicted_Transactions", "Weather", "Temperature", "Holiday_Type", "Suggestion", "Confidence"]
                st.dataframe(result_df[display_cols])

                # Visualization
                st.subheader("📊 Forecast Visualization")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # Plot 1: Bar chart with peak hours and weather
                colors = ['skyblue' if i not in top_indices else 'red' 
                         for i in range(len(result_df))]
                bars = ax1.bar(result_df["Hour"], result_df["Predicted_Transactions"], color=colors)
                ax1.set_xlabel("Hour of Day")
                ax1.set_ylabel("Predicted Transactions")
                ax1.set_title("Predicted Transactions by Hour (Peak Hours in Red)")

                # Add weather labels
                for i, bar in enumerate(bars):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f"{result_df['Weather'].iloc[i]}\n{result_df['Temperature'].iloc[i]:.1f}°C",
                            ha='center', va='bottom', rotation=0, fontsize=8)

                # Plot 2: Historical vs Predicted with weather context
                recent_actual = df.tail(12)[['hour', 'transactions', 'weather_main', 'temp']].copy()
                ax2.plot(recent_actual['hour'], recent_actual['transactions'], 
                        label='Recent Actual', color='gray', linestyle='--')
                ax2.plot(result_df['Hour'], result_df['Predicted_Transactions'], 
                        label='Predicted', color='blue')
                
                # Add weather annotations
                for i, (_, row) in enumerate(result_df.iterrows()):
                    ax2.annotate(f"{row['Weather']}\n{row['Temperature']:.1f}°C",
                               (row['Hour'], row['Predicted_Transactions']),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

                ax2.set_xlabel("Hour of Day")
                ax2.set_ylabel("Transactions")
                ax2.set_title("Recent Historical vs Predicted Transactions (with Weather)")
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Staffing Recommendations with Weather Context
                st.subheader("👥 Staffing Recommendations")
                for _, row in peak_hours.iterrows():
                    current_staff = df.iloc[-1]['staff_count']
                    weather_factor = 1.2 if row['Weather'] in ['Rain', 'Snow'] else 1.0
                    holiday_factor = 1.3 if row['Holiday_Type'] in ['Holiday', 'Weekend'] else 1.0
                    
                    if row["Predicted_Transactions"] > df['transactions'].quantile(0.75):
                        recommended_staff = int(current_staff * weather_factor * holiday_factor) + 2
                    elif row["Predicted_Transactions"] > df['transactions'].mean():
                        recommended_staff = int(current_staff * weather_factor * holiday_factor) + 1
                    else:
                        recommended_staff = current_staff
                    
                    st.info(f"""
                    🕒 **Hour {row['Hour']}:00** (Confidence: {row['Confidence']})
                    - Predicted Transactions: {row['Predicted_Transactions']:.1f}
                    - Weather: {row['Weather']} ({row['Temperature']:.1f}°C)
                    - Day Type: {row['Holiday_Type']}
                    - Current Staff: {current_staff}
                    - Recommended Staff: {recommended_staff}
                    - Action: {row['Suggestion']}
                    """)

                # Display predictions
                st.success("Predictions generated successfully!")
                
                # Display holiday information at the top
                display_holiday_info(result_df)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🌦️ Weather Impact", "📊 Analytics"])
                
                with tab1:
                    # Display predictions chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=predictions["timestamp"],
                        y=predictions["transactions"],
                        mode="lines",
                        name="Predicted Transactions",
                        line=dict(color="blue")
                    ))
                    
                    # Add peak hours markers
                    peak_hours = predictions[predictions["is_peak"] == 1]
                    fig.add_trace(go.Scatter(
                        x=peak_hours["timestamp"],
                        y=peak_hours["transactions"],
                        mode="markers",
                        name="Peak Hours",
                        marker=dict(
                            color="red",
                            size=10,
                            symbol="star"
                        )
                    ))
                    
                    fig.update_layout(
                        title="Transaction Predictions with Peak Hours",
                        xaxis_title="Time",
                        yaxis_title="Predicted Transactions",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display weather strip below the chart
                    display_weather_strip(predictions)
                
                with tab2:
                    # Display weather impact analysis
                    create_weather_impact_chart(predictions)
                    create_holiday_impact_chart(predictions)
                
                with tab3:
                    # Display comprehensive analytics dashboard
                    display_weather_holiday_dashboard(predictions)
            
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                if use_mock_data:
                    st.info("ℹ️ Using mock weather and holiday data. For real-time data, add your OpenWeatherMap API key.")
                else:
                    st.error("Please ensure you have trained the model with the latest data including weather and holiday features.")
