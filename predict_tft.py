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
raw_preds, x = model.predict(prediction_dl, mode="raw", return_x=True)

forecast = raw_preds[0].detach().cpu().numpy().flatten()
time_steps = x["decoder_time_idx"][0].detach().cpu().numpy()

# === Step 6: Identify top N peak hours ===
top_indices = forecast.argsort()[-top_n_peaks:][::-1]
peak_hours = time_steps[top_indices]

# === Step 7: Save and show results ===
result_df = pd.DataFrame({
    "Hour": time_steps,
    "Predicted Transactions": forecast
})
result_df["Suggestion"] = result_df["Predicted Transactions"].apply(
    lambda x: "📈 Add staff/stock!" if x > result_df["Predicted Transactions"].quantile(0.75) else "✅ Normal"
)

print("\n🔮 Forecasted Transactions (Next 6 hours):")
print(result_df)

print("\n🔥 Peak Hours:")
for h in peak_hours:
    print(f"🕒 Hour {int(h)}")

result_df.to_csv(output_file, index=False)
print(f"\n✅ Forecast saved to: {output_file}")



