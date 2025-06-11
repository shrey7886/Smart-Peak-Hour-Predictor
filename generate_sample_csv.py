import pandas as pd
import random

rows = []
start_time = pd.Timestamp("2024-05-01 00:00:00")

for i in range(240):  # 10 days of hourly data
    ts = start_time + pd.Timedelta(hours=i)
    transactions = random.randint(5, 50)

    promo_flag = random.choices([0, 1], weights=[0.7, 0.3])[0]
    promo_type = random.choice(["discount", "combo", "bogo", None]) if promo_flag else None

    staff = random.randint(2, 5)
    inventory_alert = random.choices([0, 1], weights=[0.9, 0.1])[0]

    event_flag = random.choices([0, 1], weights=[0.8, 0.2])[0]
    event_name = random.choice(["Festival", "Local Fair", "Closing Hour Sale", None]) if event_flag else None

    row = [ts, transactions, promo_flag, promo_type, staff, event_flag, event_name, inventory_alert]
    rows.append(row)

df = pd.DataFrame(rows, columns=[
    "timestamp", "transactions", "promotion_flag", "promotion_type",
    "staff_count", "event_flag", "event_name", "inventory_alert"
])

df.to_csv("data/shop_sample_large.csv", index=False)
print("✅ CSV generated: data/shop_sample_large.csv")
