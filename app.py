# app.py

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

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

            # Load preprocessed data
            df = pd.read_csv("data/processed_shop.csv")
            df["shop"] = "shop_1"
            df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
            df["event_name"] = df["event_name"].astype(str).fillna("None")

            max_encoder_length = 24
            max_prediction_length = 6

            dataset = TimeSeriesDataSet(
                df,
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
            )

            model_path = "models/shop_tft.ckpt"
            if not os.path.exists(model_path):
                st.error("❌ Model checkpoint not found. Please train it using Phase 2.")
                st.stop()

            # ✅ FIXED: Correct unpacking of predict() output
            model = TemporalFusionTransformer.load_from_checkpoint(
                model_path,
                map_location="cpu"
            )

            raw_preds, x, _ = model.predict(dataset, mode="raw", return_x=True)
            forecast = raw_preds[0].detach().cpu().numpy().flatten()
            time_steps = x["decoder_time_idx"][0].numpy()

            # Detect peak hours
            top_n = 3
            top_idx = forecast.argsort()[-top_n:][::-1]
            peak_hours = time_steps[top_idx]

            # Format output
            result_df = pd.DataFrame({
                "Hour": time_steps,
                "Predicted Transactions": forecast
            })
            result_df["Suggestion"] = result_df["Predicted Transactions"].apply(
                lambda x: "📈 Add staff/stock!" if x > result_df["Predicted Transactions"].quantile(0.75) else "✅ Normal"
            )

            # Show table
            st.dataframe(result_df)

            # Plot chart
            st.subheader("📊 Forecast Chart")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(result_df["Hour"], result_df["Predicted Transactions"], color="skyblue")
            for hour in peak_hours:
                ax.axvline(x=hour, color="red", linestyle="--", alpha=0.7)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Predicted Transactions")
            st.pyplot(fig)

            # Save result
            result_df.to_csv("data/predicted_peak_hours.csv", index=False)
            st.success("✅ Forecast saved to: data/predicted_peak_hours.csv")
