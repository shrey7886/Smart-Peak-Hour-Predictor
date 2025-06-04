# predict_tft.py

import os
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder

# === Config ===
processed_file = "data/processed_shop.csv"
model_checkpoint = "models/shop_tft.ckpt"
output_file = "data/predicted_peak_hours.csv"
max_encoder_length = 24
max_prediction_length = 6
top_n_peaks = 3

# === Step 1: Load processed data ===
if not os.path.exists(processed_file):
    raise FileNotFoundError("❌ 'processed_shop.csv' not found. Run preprocessing first.")

df = pd.read_csv(processed_file)
df["shop"] = "shop_1"
df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
df["event_name"] = df["event_name"].astype(str).fillna("None")

# === Step 2: Recreate TimeSeriesDataSet ===
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

# === Step 3: Load model checkpoint ===
if not os.path.exists(model_checkpoint):
    raise FileNotFoundError("❌ Model checkpoint not found. Train the model using train_tft.py.")

model = TemporalFusionTransformer.load_from_checkpoint(model_checkpoint)

# === Step 4: Forecast ===
raw_preds, x = model.predict(dataset, mode="raw", return_x=True)
forecast = raw_preds[0].detach().cpu().numpy().flatten()
time_steps = x["decoder_time_idx"][0].numpy()

# === Step 5: Identify peak hours ===
top_indices = forecast.argsort()[-top_n_peaks:][::-1]
peak_hours = time_steps[top_indices]

# === Step 6: Build result DataFrame ===
result_df = pd.DataFrame({
    "Hour": time_steps,
    "Predicted Transactions": forecast
})
result_df["Suggestion"] = result_df["Predicted Transactions"].apply(
    lambda x: "📈 Add staff/stock!" if x > result_df["Predicted Transactions"].quantile(0.75) else "✅ Normal"
)

# === Step 7: Output ===
print("🔮 Forecasted Transactions (Next 6 hours):")
print(result_df)

print("\n🔥 Top Peak Hours:")
for h in peak_hours:
    print(f"🕒 Hour {int(h)}")

result_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")
