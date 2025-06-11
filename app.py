# ✅ Final `app.py` with Synced Prediction Logic

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from utils.preprocessing import validate_and_preprocess

# ───────────── Streamlit UI Setup ─────────────
st.set_page_config(page_title="📈 Smart Peak Hour Predictor", layout="wide")
st.title("📊 Smart Peak Hour Predictor")

REQUIRED_COLUMNS = [
    'timestamp', 'transactions', 'promotion_flag', 'promotion_type',
    'staff_count', 'event_flag', 'event_name', 'inventory_alert'
]

# ───────────── Step 1: Upload CSV ─────────────
st.header("📥 Step 1: Upload Sales CSV")
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
                df = validate_and_preprocess(file_path)
                st.success("✅ Data preprocessed successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Preprocessing error: {e}")

        # ───────────── Step 2: Forecasting ─────────────
        if st.button("🔮 Predict Peak Hours"):
            st.subheader("📈 Prediction Results")
            
            try:
                # Load and validate data
                df = pd.read_csv("data/processed_shop.csv")
                st.write("Initial data shape:", df.shape)
                
                # Add shop identifier and handle categorical columns
                df["shop"] = "shop_1"
                df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
                df["event_name"] = df["event_name"].astype(str).fillna("None")

                # Define required columns and check for missing values
                required_reals = [
                    "time_idx", "transactions", "hour", "day_of_week", "is_weekend",
                    "staff_count", "promotion_flag", "event_flag", "inventory_alert"
                ]
                
                # Check for missing values before dropping
                missing_counts = df[required_reals].isnull().sum()
                if missing_counts.any():
                    st.warning(f"Found missing values before cleaning:\n{missing_counts[missing_counts > 0]}")
                
                # Drop rows with missing values and sort
                df = df.dropna(subset=required_reals)
                df = df.sort_values("time_idx")
                st.write("Data shape after cleaning:", df.shape)

                # TimeSeriesDataSet parameters
                max_encoder_length = 24
                max_prediction_length = 6
                top_n_peaks = 3

                # Validate time index
                if df["time_idx"].max() <= max_encoder_length + max_prediction_length:
                    st.error("❌ Not enough data points for prediction")
                    st.stop()

                training_cutoff = df["time_idx"].max() - max_prediction_length
                
                # Create training dataset with validation
                try:
                    training = TimeSeriesDataSet(
                        df[df["time_idx"] <= training_cutoff],
                        time_idx="time_idx",
                        target="transactions",
                        group_ids=["shop"],
                        time_varying_known_reals=[
                            "time_idx", "hour", "day_of_week", "is_weekend",
                            "staff_count", "promotion_flag", "event_flag", "inventory_alert"
                        ],
                        time_varying_unknown_reals=["transactions"],
                        time_varying_known_categoricals=["promotion_type", "event_name"],
                        categorical_encoders={
                            "promotion_type": NaNLabelEncoder(add_nan=True),
                            "event_name": NaNLabelEncoder(add_nan=True)
                        },
                        max_encoder_length=max_encoder_length,
                        max_prediction_length=max_prediction_length,
                        add_relative_time_idx=True,
                        add_target_scales=True,
                        add_encoder_length=True,
                    )
                    st.write("✅ Training dataset created successfully")
                except Exception as e:
                    st.error(f"❌ Error creating training dataset: {str(e)}")
                    st.stop()

                # Create prediction dataset
                try:
                    prediction = TimeSeriesDataSet.from_dataset(
                        training, 
                        df, 
                        predict=True, 
                        stop_randomization=True
                    )
                    prediction_dl = prediction.to_dataloader(train=False, batch_size=1, num_workers=0)
                    st.write("✅ Prediction dataset created successfully")
                except Exception as e:
                    st.error(f"❌ Error creating prediction dataset: {str(e)}")
                    st.stop()

                # Load model
                model_path = "models/shop_tft.ckpt"
                if not os.path.exists(model_path):
                    st.error("❌ Model checkpoint not found. Please train it using Phase 2.")
                    st.stop()

                try:
                    model = TemporalFusionTransformer.load_from_checkpoint(
                        model_path, 
                        map_location=torch.device("cpu")
                    )
                    st.write("✅ Model loaded successfully")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.stop()

                # Make predictions
                try:
                    predictions = model.predict(prediction_dl)
                    predictions = predictions.cpu().numpy()
                    st.write("✅ Predictions generated successfully")

                    # Get the last time index and create future time steps
                    last_time_idx = df['time_idx'].max()
                    future_time_steps = range(last_time_idx + 1, last_time_idx + max_prediction_length + 1)

                    # Create results dataframe
                    result_df = pd.DataFrame({
                        "Hour": future_time_steps,
                        "Predicted_Transactions": predictions[-max_prediction_length:].flatten()
                    })

                    # Identify peak hours
                    top_indices = result_df["Predicted_Transactions"].argsort()[-top_n_peaks:][::-1]
                    peak_hours = result_df.iloc[top_indices]["Hour"].values

                    # Add suggestions
                    result_df["Suggestion"] = result_df["Predicted_Transactions"].apply(
                        lambda x: "📈 Add staff/stock!" if x > result_df["Predicted_Transactions"].quantile(0.75) else "✅ Normal"
                    )

                    st.dataframe(result_df)

                    st.subheader("📊 Forecast Chart")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(result_df["Hour"], result_df["Predicted_Transactions"], color="skyblue")
                    for hour in peak_hours:
                        ax.axvline(x=hour, color="red", linestyle="--", alpha=0.7)
                    ax.set_xlabel("Hour")
                    ax.set_ylabel("Predicted Transactions")
                    st.pyplot(fig)

                    result_df.to_csv("data/predicted_peak_hours.csv", index=False)
                    st.success("✅ Forecast saved to: data/predicted_peak_hours.csv")

                    st.subheader("🔥 Peak Hours:")
                    for h in peak_hours:
                        st.write(f"🕒 Hour {int(h)}")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.write("Debug info:")
                    st.write("predictions shape:", predictions.shape if 'predictions' in locals() else "Not available")
                    st.stop()
                
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.stop()
