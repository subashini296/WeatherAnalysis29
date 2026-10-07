"""
Generate a realistic synthetic retail sales dataset for the
"Sales Trend Visualization" project (CodTech Data Analytics Internship).

Simulates 2 years of daily transactions across regions, categories and
products, with seasonality, weekend effects, a holiday spike, promotions,
and a mild upward growth trend + random noise -- similar to real retail data.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------
regions = ["North", "South", "East", "West"]
categories = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers"],
    "Home & Kitchen": ["Mixer", "Cookware Set", "Vacuum Cleaner", "Lamp"],
    "Beauty": ["Skincare Kit", "Perfume", "Makeup Kit", "Hair Dryer"],
    "Groceries": ["Snack Pack", "Beverages", "Organic Foods", "Dairy Pack"],
}
base_price = {
    "Smartphone": 15000, "Laptop": 42000, "Headphones": 1800, "Smartwatch": 3500,
    "T-Shirt": 500, "Jeans": 1200, "Jacket": 2500, "Sneakers": 2200,
    "Mixer": 2200, "Cookware Set": 3200, "Vacuum Cleaner": 6500, "Lamp": 900,
    "Skincare Kit": 1500, "Perfume": 2800, "Makeup Kit": 1900, "Hair Dryer": 1300,
    "Snack Pack": 250, "Beverages": 180, "Organic Foods": 600, "Dairy Pack": 220,
}

dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")

rows = []
order_id = 100000

for date in dates:
    day_of_year = date.timetuple().tm_yday
    # Overall gentle growth trend across the 2 years
    trend = 1 + 0.00035 * (date - dates[0]).days
    # Yearly seasonality (peaks around Oct-Dec festive/holiday season)
    seasonality = 1 + 0.35 * np.sin(2 * np.pi * (day_of_year - 260) / 365)
    # Weekend boost
    weekend_boost = 1.25 if date.weekday() >= 5 else 1.0
    # Festive spike windows (Diwali-ish Oct/Nov, New Year, summer sale June)
    festive_boost = 1.0
    if (date.month == 10 and 15 <= date.day <= 31) or (date.month == 11 and date.day <= 10):
        festive_boost = 1.9
    if date.month == 12 and 24 <= date.day <= 31:
        festive_boost = 2.1
    if date.month == 6 and 1 <= date.day <= 10:
        festive_boost = 1.5

    # Number of transactions today
    n_orders = int(np.random.poisson(28 * trend * seasonality * weekend_boost * festive_boost))

    for _ in range(n_orders):
        region = np.random.choice(regions, p=[0.28, 0.24, 0.22, 0.26])
        category = np.random.choice(list(categories.keys()))
        product = np.random.choice(categories[category])
        qty = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.4, 0.2, 0.15, 0.15, 0.06, 0.04])
        price = base_price[product] * np.random.uniform(0.92, 1.08)
        discount_pct = np.random.choice([0, 0, 5, 10, 15, 20], p=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05])
        revenue = round(price * qty * (1 - discount_pct / 100), 2)

        rows.append([
            order_id, date.date().isoformat(), region, category, product,
            qty, round(price, 2), discount_pct, revenue
        ])
        order_id += 1

df = pd.DataFrame(rows, columns=[
    "OrderID", "OrderDate", "Region", "Category", "Product",
    "Quantity", "UnitPrice", "DiscountPct", "Revenue"
])

df.to_csv("/home/claude/sales_project/data/sales_data.csv", index=False)
print("Rows generated:", len(df))
print(df.head())
print(df["OrderDate"].min(), "to", df["OrderDate"].max())
