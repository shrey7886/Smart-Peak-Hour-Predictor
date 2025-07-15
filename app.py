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
import requests  # Add this import

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
                    st.experimental_rerun()
                
                if st.button("📈 View History", key=f"history_{shop['shop_id']}"):
                    st.session_state.current_shop = shop['shop_id']
                    st.session_state.show_history = True
                    initialize_managers(shop['shop_id'])
                    st.experimental_rerun()
    
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
                st.experimental_rerun()

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
