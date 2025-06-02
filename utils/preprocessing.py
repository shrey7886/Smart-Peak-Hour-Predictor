import pandas as pd

REQUIRED_COLUMNS = [
    'timestamp', 'transactions', 'promotion_flag', 'promotion_type',
    'staff_count', 'event_flag', 'event_name', 'inventory_alert'
]

def validate_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    # Validate columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

    # Parse timestamp and add features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['time_idx'] = range(len(df))

    df.to_csv("data/processed_shop.csv", index=False)
    return df
