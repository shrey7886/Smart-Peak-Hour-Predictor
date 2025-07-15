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
    body, .main, .stApp {
        background-color: #181c20 !important;
        color: #f5f6fa !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stSidebar, .css-1d391kg, .css-1lcbmhc, .css-1v0mbdj, .css-1cypcdb {
        background-color: #20232a !important;
        color: #f5f6fa !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        background: #2563eb !important;
        color: #fff !important;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 8px rgba(30,41,59,0.08);
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: #1e293b !important;
        color: #fff !important;
    }
    .stProgress .st-bo {
        background-color: #2563eb;
    }
    .metric-card, .info-box, .warning-box, .success-box {
        background-color: #23272f;
        color: #f5f6fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(30,41,59,0.10);
    }
    .info-box {
        border-left: 4px solid #2563eb;
    }
    .warning-box {
        border-left: 4px solid #facc15;
        background-color: #3a2e1a;
        color: #facc15;
    }
    .success-box {
        border-left: 4px solid #22c55e;
        background-color: #1e2d23;
        color: #22c55e;
    }
    .stMarkdown, .stTextInput>div>input, .stSelectbox>div>div>div>input, .stFileUploader>div>div>div>input {
        color: #f5f6fa !important;
        background-color: #23272f !important;
        border-radius: 6px;
    }
    .stDownloadButton>button {
        background: #2563eb;
        color: #fff;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 8px rgba(30,41,59,0.08);
        transition: background 0.2s;
    }
    .stDownloadButton>button:hover {
        background: #1e293b;
        color: #fff;
    }
    .stFileUploader>div>div {
        background: #23272f !important;
        color: #f5f6fa !important;
        border-radius: 6px;
        border: 1px solid #2563eb;
    }
    .stMetric {
        background: #23272f;
        color: #f5f6fa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(30,41,59,0.10);
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #23272f;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #f5f6fa;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb;
        color: #fff;
        border-radius: 8px 8px 0 0;
    }
    .stAlert {
        background: #23272f !important;
        color: #f5f6fa !important;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
    }
    .stSpinner {
        color: #2563eb !important;
    }
    .stTextInput>div>input:focus, .stSelectbox>div>div>div>input:focus {
        border: 1.5px solid #2563eb !important;
        outline: none !important;
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
                    st.session_state.shop_selected = True
                    initialize_managers(shop['shop_id'])
                    st.experimental_rerun()
                
                if st.button("📈 View History", key=f"history_{shop['shop_id']}"):
                    st.session_state.current_shop = shop['shop_id']
                    st.session_state.shop_selected = True
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
                st.session_state.shop_selected = True
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

        # --- Sidebar Navigation ---
        st.sidebar.title("Smart Peak Hour Predictor")
        st.sidebar.markdown("---")

        # Shop Management Section
        st.sidebar.header("🏪 Shop Management")
        search_query = st.sidebar.text_input("🔍 Search Shops", "")
        all_shops = get_all_shops()
        if search_query:
            all_shops = [shop for shop in all_shops if search_query.lower() in shop['shop_id'].lower()]
        st.sidebar.metric("Total Shops", len(all_shops))
        active_shops = len([shop for shop in all_shops if 'last_upload' in shop and shop['last_upload']])
        st.sidebar.metric("Active Shops", active_shops)
        if all_shops:
            st.sidebar.markdown("### 📋 Existing Shops")
            for shop in all_shops:
                if st.sidebar.button(f"Select {shop['shop_id']}", key=f"select_{shop['shop_id']}"):
                    st.session_state.current_shop = shop['shop_id']
                    st.session_state.shop_selected = True
                    initialize_managers(shop['shop_id'])
                    st.experimental_rerun()
        st.sidebar.markdown("---")

        # Data Upload Section
        st.sidebar.header("📤 Data Upload")
        if st.session_state.get('current_shop'):
            uploaded_file = st.sidebar.file_uploader("Upload CSV for this shop", type=["csv"], key="sidebar_upload")
            if uploaded_file is not None:
                try:
                    df_preview = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df_preview
                    st.sidebar.success("File uploaded! Preview below.")
                    st.sidebar.dataframe(df_preview.head(10))
                    # Save uploaded data
                    shop_data_path = f"data/shops/{st.session_state.current_shop}/processed_data.csv"
                    os.makedirs(os.path.dirname(shop_data_path), exist_ok=True)
                    df_preview.to_csv(shop_data_path, index=False)
                    st.session_state.shop_data = df_preview
                except Exception as upload_err:
                    st.sidebar.error(f"Failed to process uploaded file: {upload_err}")
        else:
            st.sidebar.info("Select a shop to enable data upload.")
        st.sidebar.markdown("---")

        # Model Selection Section
        st.sidebar.header("🤖 Model Selection")
        model_options = ["Temporal Fusion Transformer"]  # Add more as needed
        selected_model = st.sidebar.selectbox("Choose Prediction Model", model_options)
        st.sidebar.markdown("---")

        # Results/Analytics Section
        st.sidebar.header("📊 Results & Analytics")
        st.sidebar.info("Results and analytics will appear in the main area after processing.")
        st.sidebar.markdown("---")

        # History Section
        st.sidebar.header("🕑 History")
        st.sidebar.info("Recent uploads and predictions will be tracked here.")
        st.sidebar.markdown("---")

        # Settings Section
        st.sidebar.header("⚙️ Settings")
        st.sidebar.info("User preferences and configuration options coming soon.")
        st.sidebar.markdown("---")

        # --- Chatbot Section ---
        st.sidebar.header("💬 Chatbot")
        st.sidebar.markdown("Ask questions about the app, data, or features!")
        # API key input
        hf_api_key = st.sidebar.text_input(
            "Hugging Face API Key",
            type="password",
            help="Enter your Hugging Face Inference API key. Get one at https://huggingface.co/settings/tokens"
        )
        if hf_api_key:
            st.session_state["hf_api_key"] = hf_api_key
        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        for msg in st.session_state["chat_history"][-6:]:
            role = "🧑" if msg["role"] == "user" else "🤖"
            st.sidebar.markdown(f"**{role}:** {msg['content']}")
        # User input
        user_input = st.sidebar.text_input("Type your question:", key="chatbot_input")
        if st.sidebar.button("Send", key="chatbot_send"):
            if user_input:
                st.session_state["chat_history"].append({"role": "user", "content": user_input})
                # Placeholder for bot response (to be replaced with RAG logic)
                st.session_state["chat_history"].append({"role": "assistant", "content": "(Thinking... will answer soon)"})

        # --- Main Page Content ---
        st.title("Smart Peak Hour Predictor Dashboard")
        if not st.session_state.get('current_shop'):
            st.markdown("""
                # 👋 Welcome!
                Select a shop from the sidebar or add a new one to get started.
            """)
            st.stop()

        # Data Preview and Validation
        if st.session_state.get('shop_data') is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.shop_data.head(20))
            st.write(f"Rows: {st.session_state.shop_data.shape[0]}, Columns: {st.session_state.shop_data.shape[1]}")
            # Basic validation
            required_cols = ['timestamp', 'transactions']
            missing_cols = [col for col in required_cols if col not in st.session_state.shop_data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()
        else:
            st.info("No data uploaded for this shop. Please upload a CSV file using the sidebar.")
            st.stop()

        # Model Training and Prediction
        model_path = "models/shop_tft.ckpt"
        if not os.path.exists(model_path):
            st.warning("⚠️ Model not found. Please train the model first.")
            if st.button("Train Model", key="train_model_btn_main"):
                with st.spinner("Training model. This may take a while..."):
                    try:
                        import subprocess
                        result = subprocess.run(["python", "train_tft.py"], capture_output=True, text=True)
                        if result.returncode == 0:
                            st.success("Model trained successfully! You can now run predictions.")
                            st.session_state.model_trained = True
                        else:
                            st.error(f"Training failed: {result.stderr}")
                    except Exception as train_err:
                        st.error(f"Error during training: {train_err}")
            # Only show dashboard in preview mode (no predictions)
            display_shop_dashboard_v2(
                df=st.session_state.shop_data,
                shop_id=st.session_state.current_shop,
                shop_name=st.session_state.current_shop,
                skip_predictions=True
            )
        else:
            if st.button("Predict", key="predict_btn_main"):
                st.session_state.run_predictions = True
            if st.session_state.get("run_predictions"):
                display_shop_dashboard_v2(
                    df=st.session_state.shop_data,
                    shop_id=st.session_state.current_shop,
                    shop_name=st.session_state.current_shop,
                    skip_predictions=False
                )
            else:
                st.info("Click 'Predict' to run predictions and view analytics.")

    except Exception as e:
        logger.error(f"Error in main app: {str(e)}")
        st.error("An error occurred. Please check the logs.")

if __name__ == "__main__":
    main()
