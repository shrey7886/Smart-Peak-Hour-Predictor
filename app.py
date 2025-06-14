# ✅ Final `app.py` with Enhanced Analytics

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime, timedelta

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
                
                # Show data insights
                st.success("✅ Data preprocessed successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
                with col3:
                    st.metric("Average Daily Transactions", f"{df['transactions'].mean():.1f}")

                # Show sample data
                st.subheader("📋 Sample Data")
                st.dataframe(df.head())

                # Show data distribution
                st.subheader("📊 Data Distribution")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Transactions distribution
                sns.histplot(data=df, x='transactions', bins=30, ax=ax1)
                ax1.set_title('Transaction Distribution')
                
                # Transactions by hour
                hourly_avg = df.groupby('hour')['transactions'].mean()
                sns.barplot(x=hourly_avg.index, y=hourly_avg.values, ax=ax2)
                ax2.set_title('Average Transactions by Hour')
                ax2.set_xlabel('Hour of Day')
                ax2.set_ylabel('Average Transactions')
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Preprocessing error: {e}")

        # ───────────── Step 2: Forecasting ─────────────
        if st.button("🔮 Predict Peak Hours"):
            st.subheader("📈 Prediction Results")

            try:
                # Load and validate data
                df = pd.read_csv("data/processed_shop.csv")
                
                # Add shop identifier and handle categorical columns
                df["shop"] = "shop_1"
                df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
                df["event_name"] = df["event_name"].astype(str).fillna("None")

                # Define required columns and check for missing values
                required_reals = [
                    "time_idx", "transactions", "hour", "day_of_week", "is_weekend",
                    "staff_count", "promotion_flag", "event_flag", "inventory_alert"
                ]
                
                # Drop rows with missing values and sort
                df = df.dropna(subset=required_reals)
                df = df.sort_values("time_idx")

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
                        "Predicted_Transactions": predictions[-max_prediction_length:].flatten()
                    })

                # Identify peak hours
                top_indices = result_df["Predicted_Transactions"].argsort()[-top_n_peaks:][::-1]
                peak_hours = result_df.iloc[top_indices].copy()  # Create a copy to avoid SettingWithCopyWarning

                # Add suggestions and confidence levels
                result_df["Suggestion"] = result_df["Predicted_Transactions"].apply(
                    lambda x: "📈 Add staff/stock!" if x > result_df["Predicted_Transactions"].quantile(0.75) else "✅ Normal"
                )

                # Calculate confidence levels based on prediction values
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
                col1, col2, col3 = st.columns(3)
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
                    busy_hours = len(result_df[result_df["Suggestion"] == "📈 Add staff/stock!"])
                    st.metric("Busy Hours Ahead", 
                            f"{busy_hours}",
                            f"{(busy_hours/len(result_df))*100:.0f}% of predicted period")

                # Display detailed predictions
                st.subheader("🕒 Hourly Predictions")
                st.dataframe(result_df[["Timestamp", "Hour", "Predicted_Transactions", "Suggestion", "Confidence"]])

                # Visualization
                st.subheader("📊 Forecast Visualization")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                # Plot 1: Bar chart with peak hours highlighted
                colors = ['skyblue' if i not in top_indices else 'red' 
                         for i in range(len(result_df))]
                ax1.bar(result_df["Hour"], result_df["Predicted_Transactions"], color=colors)
                ax1.set_xlabel("Hour of Day")
                ax1.set_ylabel("Predicted Transactions")
                ax1.set_title("Predicted Transactions by Hour (Peak Hours in Red)")

                # Plot 2: Historical vs Predicted
                recent_actual = df.tail(12)[['hour', 'transactions']].copy()
                ax2.plot(recent_actual['hour'], recent_actual['transactions'], 
                        label='Recent Actual', color='gray', linestyle='--')
                ax2.plot(result_df['Hour'], result_df['Predicted_Transactions'], 
                        label='Predicted', color='blue')
                ax2.set_xlabel("Hour of Day")
                ax2.set_ylabel("Transactions")
                ax2.set_title("Recent Historical vs Predicted Transactions")
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Staffing Recommendations
                st.subheader("👥 Staffing Recommendations")
                for idx, row in peak_hours.iterrows():
                    current_staff = df.iloc[-1]['staff_count']
                    if row["Predicted_Transactions"] > df['transactions'].quantile(0.75):
                        recommended_staff = current_staff + 2
                    elif row["Predicted_Transactions"] > df['transactions'].mean():
                        recommended_staff = current_staff + 1
                    else:
                        recommended_staff = current_staff
                    
                    st.info(f"""
                    🕒 **Hour {row['Hour']}:00** (Confidence: {row['Confidence']})
                    - Predicted Transactions: {row['Predicted_Transactions']:.1f}
                    - Current Staff: {current_staff}
                    - Recommended Staff: {recommended_staff}
                    - Action: {row['Suggestion']}
                    """)

                # Save predictions
                result_df.to_csv("data/predicted_peak_hours.csv", index=False)
                st.success("✅ Detailed forecast saved to: data/predicted_peak_hours.csv")
            
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.stop()
