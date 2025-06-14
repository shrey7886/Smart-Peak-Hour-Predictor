import os
import shutil
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# === CONFIG ===
data_path = "data/processed_shop.csv"
model_dir = "models"
model_name = "shop_tft.ckpt"
max_encoder_length = 24
max_prediction_length = 6
batch_size = 32
min_required_rows = 30
seed = 42

# === Ensure data exists ===
if not os.path.exists(data_path):
    raise FileNotFoundError("❌ 'data/processed_shop.csv' not found. Run Phase 1 to generate it.")

# === Set seed and load data ===
seed_everything(seed)
df = pd.read_csv(data_path)
print(f"🔍 Loaded dataset with {len(df)} rows.")

if len(df) < min_required_rows:
    raise ValueError(f"❌ Dataset too small. Needs at least {min_required_rows} rows, found {len(df)}.")

# === Prepare Data ===
df["shop"] = "shop_1"
df["promotion_type"] = df["promotion_type"].astype(str).fillna("None")
df["event_name"] = df["event_name"].astype(str).fillna("None")
df = df.sort_values("time_idx")

# === Time cutoff for encoder/decoder window ===
training_cutoff = df["time_idx"].max() - max_prediction_length

# === Define Training Dataset ===
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

# === Validation Set ===
val_dataset = TimeSeriesDataSet.from_dataset(training, df, stop_randomization=True)

# === Dataloaders ===
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# === Define Model ===
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=4,
    log_gradient_flow=False,
)

# === Checkpointing ===
checkpoint_callback = ModelCheckpoint(
    dirpath=model_dir,
    filename=model_name.replace(".ckpt", ""),
    monitor="val_loss",
    save_top_k=1,
    mode="min"
)

# === Train ===
trainer = Trainer(
    max_epochs=20,
    accelerator="auto",
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
)

print(f"📚 Starting training on {len(train_dataloader)} batches...")
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# === Save Best Checkpoint as shop_tft.ckpt ===
final_path = os.path.join(model_dir, model_name)
best_model_path = checkpoint_callback.best_model_path

if best_model_path != final_path:
    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.move(best_model_path, final_path)

print(f"✅ Training complete. Model saved to: {final_path}")

