import streamlit as st
import os
import pandas as pd
from utils.preprocessing import validate_and_preprocess

st.set_page_config(page_title="Smart Peak Hour Predictor", layout="wide")
st.title("📈 Smart Peak Hour Predictor")
st.subheader("📤 Upload and Preprocess Your Sales Data")

REQUIRED_COLUMNS = [
    'timestamp', 'transactions', 'promotion_flag', 'promotion_type',
    'staff_count', 'event_flag', 'event_name', 'inventory_alert'
]

# Template download
with st.expander("📥 Download CSV Template"):
    with open("data/shop_template.csv", "r") as f:
        st.download_button(
            label="Download CSV Template",
            data=f,
            file_name="shop_template.csv",
            mime="text/csv"
        )
    st.markdown("This file shows the required columns for prediction.")

# File upload
uploaded_file = st.file_uploader("Upload your hourly sales CSV", type="csv")

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", "uploaded_shop.csv")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded!")

    df_check = pd.read_csv(file_path, nrows=1)
    uploaded_columns = list(df_check.columns)
    missing = [col for col in REQUIRED_COLUMNS if col not in uploaded_columns]
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
    else:
        if st.button("🚀 Preprocess Data"):
            try:
                df = validate_and_preprocess(file_path)
                st.success("✅ Data preprocessed successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error: {e}")
