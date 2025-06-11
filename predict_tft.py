import os
import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder

# === CONFIG ===
processed_file = "data/processed_shop.csv"
model_checkpoint = "models/shop_tft.ckpt"
output_file = "data/predicted_peak_hours.csv"
max_encoder_length = 24
max_prediction_length = 6
top_n_peaks = 3

# === Step 1: Load and clean processed data ===
if not os.path.exists(processed_file):
    raise FileNotFoundError("❌ 'processed_shop.csv' not found. Run preprocessing first.")

df = pd.read_csv(processed_file)

# Clean categorical columns
df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
df["event_name"] = df["event_name"].astype(str).fillna("None")
df["shop"] = "shop_1"
df = df.sort_values("time_idx")

# Drop missing critical features
required_columns = [
    "time_idx", "transactions", "hour", "day_of_week", "is_weekend",
    "staff_count", "promotion_flag", "event_flag", "inventory_alert"
]
df = df.dropna(subset=required_columns)

# === Step 2: Define TimeSeriesDataSet for training context ===
training_cutoff = df["time_idx"].max() - max_prediction_length

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

# === Step 3: Create prediction dataset ===
prediction = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
prediction_dl = prediction.to_dataloader(train=False, batch_size=1, num_workers=0)

# === Step 4: Load model checkpoint ===
if not os.path.exists(model_checkpoint):
    raise FileNotFoundError("❌ Model checkpoint not found. Train the model first.")

model = TemporalFusionTransformer.load_from_checkpoint(model_checkpoint, map_location=torch.device("cpu"))

# === Step 5: Make predictions ===
predictions = model.predict(prediction_dl)
predictions = predictions.cpu().numpy()

# Get the last time index and create future time steps
last_time_idx = df['time_idx'].max()
future_time_steps = range(last_time_idx + 1, last_time_idx + max_prediction_length + 1)

# === Step 6: Create results dataframe ===
result_df = pd.DataFrame({
    "Hour": future_time_steps,
    "Predicted_Transactions": predictions[-max_prediction_length:].flatten()
})

# === Step 7: Identify peak hours ===
top_indices = result_df["Predicted_Transactions"].argsort()[-top_n_peaks:][::-1]
peak_hours = result_df.iloc[top_indices]["Hour"].values

# Add suggestions
result_df["Suggestion"] = result_df["Predicted_Transactions"].apply(
    lambda x: "📈 Add staff/stock!" if x > result_df["Predicted_Transactions"].quantile(0.75) else "✅ Normal"
)

print("\n🔮 Forecasted Transactions (Next 6 hours):")
print(result_df)

print("\n🔥 Peak Hours:")
for h in peak_hours:
    print(f"🕒 Hour {int(h)}")

result_df.to_csv(output_file, index=False)
print(f"\n✅ Forecast saved to: {output_file}")
