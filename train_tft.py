import os
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# 🚨 Ensure the processed data exists
if not os.path.exists("data/processed_shop.csv"):
    raise FileNotFoundError("❌ 'data/processed_shop.csv' not found. Run Phase 1 to generate it.")

# 📊 Load and prepare data
df = pd.read_csv("data/processed_shop.csv")
print(f"🔍 Loaded dataset with {len(df)} rows.")

# 🧠 Add required group column
df["shop"] = "shop_1"

# 🧹 Clean categorical columns
df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
df["event_name"] = df["event_name"].astype(str).fillna("None")

# ✅ Minimum rows check
min_required_rows = 30  # min rows needed: encoder + prediction
if len(df) < min_required_rows:
    raise ValueError(f"❌ Dataset too small. Needs at least {min_required_rows} rows, found {len(df)}.")

# ⚙️ Model configs for small dataset
max_encoder_length = 24  # 1 day
max_prediction_length = 6  # 6 hours

# ✅ Create TimeSeriesDataSet
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

# 🔁 Dataloader
train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)

# 🧠 TFT Model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=torch.nn.MSELoss(),  # Replace with QuantileLoss later for quantile forecasting
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=4,
)

# 💾 Save best model checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath="models",
    filename="shop_tft",
    monitor="train_loss",
    save_top_k=1,
    mode="min"
)

# ⚡ Train the model
trainer = Trainer(
    max_epochs=20,
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback],
    enable_model_summary=True,
    log_every_n_steps=10,
    accelerator="auto"
)

# 🚀 Fit model
trainer.fit(tft, train_dataloaders=train_dataloader)

# ✅ Done
print("✅ Model training complete. Checkpoint saved at: models/shop_tft.ckpt")
