import os
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
import numpy as np

# --- Configuration ---
MODEL_PATH = "models/shop_tft.ckpt"
HISTORICAL_DATA_PATH = "data/processed_shop.csv"
MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 6 # This should match the model's training

# --- FastAPI App Setup ---
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    try:
        future_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    if not isinstance(future_data, list):
        raise HTTPException(status_code=400, detail="Payload must be a list of future data objects.")

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"Model checkpoint not found at {MODEL_PATH}")

    if not os.path.exists(HISTORICAL_DATA_PATH):
        raise HTTPException(status_code=500, detail=f"Historical data not found at {HISTORICAL_DATA_PATH}")

    # Load historical data to provide context for the model
    historical_df = pd.read_csv(HISTORICAL_DATA_PATH)
    
    # Ensure we have enough historical data (at least max_encoder_length + some buffer)
    if len(historical_df) < MAX_ENCODER_LENGTH + 10:
        raise HTTPException(status_code=500, detail=f"Not enough historical data. Need at least {MAX_ENCODER_LENGTH + 10} rows, found {len(historical_df)}")
    
    # Take the last portion of historical data as context
    encoder_data = historical_df.tail(MAX_ENCODER_LENGTH + 10).copy()
    
    # Create a DataFrame for the future data
    future_df = pd.DataFrame(future_data)
    
    # Ensure future data has all required columns
    required_columns = [
        "timestamp", "hour", "day_of_week", "is_weekend", "staff_count", 
        "promotion_flag", "promotion_type", "event_flag", "event_name", 
        "inventory_alert", "temp", "humidity", "rain", "snow", "wind_speed", 
        "clouds", "is_holiday", "holiday_type", "holiday_name", "weather_main"
    ]
    
    for col in required_columns:
        if col not in future_df.columns:
            if col in ["promotion_type", "event_name", "weather_main", "holiday_type", "holiday_name"]:
                future_df[col] = "None"
            elif col in ["promotion_flag", "event_flag", "inventory_alert", "is_holiday", "is_weekend"]:
                future_df[col] = 0
            else:
                future_df[col] = 0.0

    # --- Preprocessing for both encoder and future data ---
    # Combine historical and future data
    combined_df = pd.concat([encoder_data, future_df], ignore_index=True)
    
    # Convert timestamp to datetime
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
    
    # Create time_idx - use the existing time_idx from historical data and continue for future data
    last_time_idx = encoder_data["time_idx"].max()
    future_time_indices = range(last_time_idx + 1, last_time_idx + 1 + len(future_df))
    
    # Fill NaN values in time_idx with future time indices
    combined_df.loc[len(encoder_data):, "time_idx"] = future_time_indices
    
    # Ensure time_idx is integer type for all rows (after filling NaN values)
    combined_df["time_idx"] = combined_df["time_idx"].astype(int)
    
    # Fill missing categorical values
    for col in ["promotion_type", "event_name", "weather_main", "holiday_type", "holiday_name"]:
        combined_df[col] = combined_df[col].astype(str).fillna("None")

    # Add placeholder transactions for future data (will be ignored during prediction)
    combined_df.loc[len(encoder_data):, "transactions"] = 0

    combined_df["shop"] = "shop_1"  # Assuming a single shop
    combined_df = combined_df.sort_values("time_idx").reset_index(drop=True)

    # --- Create TimeSeriesDataSet ---
    # Use the same parameters as the training
    prediction_dataset = TimeSeriesDataSet(
        combined_df,
        time_idx="time_idx",
        target="transactions",
        group_ids=["shop"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=[
            "time_idx", "hour", "day_of_week", "is_weekend",
            "staff_count", "promotion_flag", "event_flag", "inventory_alert",
            "temp", "humidity", "rain", "snow", "wind_speed", "clouds",
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
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    # --- Load Model and Predict ---
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH, map_location=torch.device("cpu"))
    
    # Create a dataloader for prediction
    prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1)
    
    # Generate predictions
    raw_predictions = model.predict(prediction_dataloader)
    
    # Process predictions
    predictions = []
    predicted_values = raw_predictions[0].numpy()
    
    # Get predictions for the future data points
    for i, future_point in enumerate(future_data):
        if i < len(predicted_values):
            predictions.append({
                "timestamp": future_point["timestamp"],
                "predicted_transactions": float(predicted_values[i])
            })

    return JSONResponse(content=predictions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 