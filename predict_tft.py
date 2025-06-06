import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

# Load the processed data
df = pd.read_csv("data/processed_shop.csv")
print(f"🔍 Loaded dataset with {len(df)} rows.")

# Add required group column
df["shop"] = "shop_1"

# Clean categorical columns
df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
df["event_name"] = df["event_name"].astype(str).fillna("None")

# ⚙️ Configuration
max_encoder_length = 24  # 1 day (matching training config)
max_prediction_length = 6  # 6 hours (matching training config)

# Create training dataset first (needed for proper data scaling)
training = TimeSeriesDataSet(
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

# Create prediction dataset
predict_dataset = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
predict_loader = predict_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

# Load the best model
print("🔄 Loading best model from checkpoint...")
best_model_path = "models/shop_tft.ckpt"
model = TemporalFusionTransformer.load_from_checkpoint(
    best_model_path,
    map_location=torch.device('cpu')
)

# Make predictions
print("🔮 Making predictions...")
predictions = model.predict(predict_loader)
predictions = predictions.cpu().numpy()

# Create results dataframe
last_time_idx = df['time_idx'].max()
future_time_idx = range(last_time_idx + 1, last_time_idx + max_prediction_length + 1)
results = pd.DataFrame({
    "time_idx": list(future_time_idx),
    "predicted_transactions": predictions[-max_prediction_length:].flatten()
})

# Save predictions
output_file = "data/predictions.csv"
results.to_csv(output_file, index=False)
print(f"✅ Predictions saved to {output_file}")

# Print summary
print("\n📊 Prediction Summary:")
print(f"Number of predictions: {len(results)}")
print(f"Prediction range: {results['predicted_transactions'].min():.2f} to {results['predicted_transactions'].max():.2f}")
print(f"Time index range: {results['time_idx'].min()} to {results['time_idx'].max()}")
