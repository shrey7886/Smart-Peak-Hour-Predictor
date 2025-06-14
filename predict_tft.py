import os
import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder

def predict_peak_hours(df, model_path="models/shop_tft.ckpt", location=None, max_encoder_length=24, max_prediction_length=6, top_n_peaks=3):
    """
    Generate predictions for peak hours using the TFT model.
    
    Args:
        df (pd.DataFrame): Input DataFrame with required features
        model_path (str): Path to the model checkpoint
        location (dict): Location information for weather/holiday features
        max_encoder_length (int): Maximum encoder length for the model
        max_prediction_length (int): Number of hours to predict
        top_n_peaks (int): Number of peak hours to identify
    
    Returns:
        pd.DataFrame: DataFrame with predictions and peak hours
    """
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Clean categorical columns
    df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
    df["event_name"] = df["event_name"].astype(str).fillna("None")
    df["weather_main"] = df["weather_main"].astype(str).fillna("Clear")
    df["shop"] = "shop_1"
    df = df.sort_values("time_idx")

    # Drop missing critical features
    required_columns = [
        "time_idx", "transactions", "hour", "day_of_week", "is_weekend",
        "staff_count", "promotion_flag", "event_flag", "inventory_alert",
        # Weather features
        "temp", "humidity", "rain", "snow", "wind_speed", "clouds",
        # Holiday features
        "is_holiday", "holiday_type", "holiday_name"
    ]
    df = df.dropna(subset=required_columns)

    # Define TimeSeriesDataSet for training context
    training_cutoff = df["time_idx"].max() - max_prediction_length

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
            "promotion_type", "event_name", "weather_main", "holiday_type", "holiday_name"
        ],
        categorical_encoders={
            "promotion_type": NaNLabelEncoder(add_nan=True),
            "event_name": NaNLabelEncoder(add_nan=True),
            "weather_main": NaNLabelEncoder(add_nan=True),
            "holiday_type": NaNLabelEncoder(add_nan=True),
            "holiday_name": NaNLabelEncoder(add_nan=True)
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

    # Load model checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model checkpoint not found at {model_path}. Train the model first.")

    model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=torch.device("cpu"))

    # Make predictions
    predictions = model.predict(prediction_dl)
    predictions = predictions.cpu().numpy()

    # Get the last time index and create future time steps
    last_time_idx = df['time_idx'].max()
    future_time_steps = range(last_time_idx + 1, last_time_idx + max_prediction_length + 1)

    # Create future timestamps
    last_timestamp = df["timestamp"].iloc[-1]
    future_timestamps = pd.date_range(start=last_timestamp, periods=max_prediction_length, freq="H")

    # Get the last max_prediction_length rows for features
    last_rows = df.iloc[-max_prediction_length:].copy()
    last_rows["timestamp"] = future_timestamps
    last_rows["time_idx"] = future_time_steps
    last_rows["transactions"] = predictions[-max_prediction_length:].flatten()

    # Create results dataframe with all required columns
    result_df = last_rows[[
        "time_idx", "timestamp", "transactions",
        "weather_main", "holiday_type", "holiday_name", "is_holiday",
        "temp", "humidity", "rain", "snow", "wind_speed", "clouds",
        "hour", "day_of_week", "is_weekend"
    ]].copy()

    # Identify peak hours
    top_indices = result_df["transactions"].argsort()[-top_n_peaks:][::-1]
    peak_hours = result_df.iloc[top_indices]

    # Add suggestions with weather and holiday context
    def get_suggestion(row):
        base_suggestion = "📈 Add staff/stock!" if row["transactions"] > result_df["transactions"].quantile(0.75) else "✅ Normal"
        weather_context = f" ({row['weather_main']}, {row['temp']:.1f}°C)"
        holiday_context = f" - {row['holiday_type']}"
        if row["holiday_name"] != "Not a holiday":
            holiday_context += f" ({row['holiday_name']})"
        return base_suggestion + weather_context + holiday_context

    result_df["suggestion"] = result_df.apply(get_suggestion, axis=1)
    result_df["is_peak"] = result_df.index.isin(top_indices).astype(int)

    return result_df

if __name__ == "__main__":
    # Example usage when running the script directly
    processed_file = "data/processed_shop.csv"
    model_checkpoint = "models/shop_tft.ckpt"
    output_file = "data/predicted_peak_hours.csv"
    
    if not os.path.exists(processed_file):
        raise FileNotFoundError("❌ 'processed_shop.csv' not found. Run preprocessing first.")
    
    df = pd.read_csv(processed_file)
    result_df = predict_peak_hours(df, model_path=model_checkpoint)
    
    print("\n🔮 Forecasted Transactions (Next 6 hours):")
    print(result_df[["time_idx", "transactions", "weather_main", "holiday_type", "suggestion"]])
    
    print("\n🔥 Peak Hours:")
    peak_hours = result_df[result_df["is_peak"] == 1]
    for _, row in peak_hours.iterrows():
        print(f"🕒 Hour {int(row['time_idx'])} - {row['weather_main']}, {row['holiday_type']}")
    
    result_df.to_csv(output_file, index=False)
    print(f"\n✅ Forecast saved to: {output_file}")
