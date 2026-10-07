"""
WEATHER PATTERN ANALYSIS - Data Analysis Internship Project
Steps: Data Load -> Cleaning -> Feature Engineering -> EDA -> Seasonal/Trend
       -> Correlation -> Extreme Events -> Rain Prediction (ML) -> Insights

Run:  python weather_analysis.py
Optional: put your own CSV as weather.csv with columns
          date, temp_max, temp_min, humidity, rainfall, wind_speed, pressure
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")
OUT = "output"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)


# ---------------------------------------------------------------
# 1. DATA LOADING (real CSV if present, else simulated dataset)
# ---------------------------------------------------------------
def generate_data():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    t = np.arange(n)
    warming = 0.0004 * t  # slow warming trend

    temp_max = 33 + 4 * np.sin(2 * np.pi * (doy - 30) / 365) + warming + rng.normal(0, 1.5, n)
    temp_min = temp_max - 9 + rng.normal(0, 1.2, n)
    # monsoon peaks (SW: Jun-Sep, NE: Oct-Dec)
    monsoon = (np.exp(-((doy - 200) ** 2) / 2500) * 0.6 + np.exp(-((doy - 300) ** 2) / 1800) * 0.9)
    humidity = np.clip(55 + 35 * monsoon + rng.normal(0, 7, n), 20, 100)
    rain_prob = np.clip(0.04 + 0.75 * monsoon + (humidity - 60) / 300, 0.01, 0.95)
    rain_flag = rng.random(n) < rain_prob
    rainfall = np.where(rain_flag, rng.gamma(1.6, 9, n), 0)
    wind = np.clip(10 + 6 * np.exp(-((doy - 190) ** 2) / 3000) + rng.normal(0, 3, n), 0, None)
    pressure = 1010 - 4 * monsoon + rng.normal(0, 2, n)

    df = pd.DataFrame({"date": dates, "temp_max": temp_max, "temp_min": temp_min,
                       "humidity": humidity, "rainfall": rainfall,
                       "wind_speed": wind, "pressure": pressure})
    # inject dirty data like real life
    for c in ["temp_max", "humidity", "wind_speed"]:
        df.loc[rng.choice(n, 40, replace=False), c] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "temp_max"] = 75  # outliers
    df = pd.concat([df, df.sample(10, random_state=1)])      # duplicates
    return df


if os.path.exists("weather.csv"):
    df = pd.read_csv("weather.csv")
    print("Loaded weather.csv")
else:
    df = generate_data()
    df.to_csv(os.path.join(OUT, "weather_raw.csv"), index=False)
    print("weather.csv not found -> simulated dataset generated")

print("\nShape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())


# ---------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------
df["date"] = pd.to_datetime(df["date"])
df = df.drop_duplicates().sort_values("date").reset_index(drop=True)

# outliers: temp_max > 50 is impossible for this region
df.loc[df["temp_max"] > 50, "temp_max"] = np.nan

# fill missing values by time interpolation
df = df.set_index("date")
num_cols = ["temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "pressure"]
df[num_cols] = df[num_cols].interpolate(method="time").bfill().ffill()
df = df.reset_index()
print("\nAfter cleaning ->", df.shape, "| missing:", int(df.isna().sum().sum()))


# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")
df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
df["temp_range"] = df["temp_max"] - df["temp_min"]
df["is_rainy"] = (df["rainfall"] >= 2.5).astype(int)


def season(m):
    if m in (3, 4, 5): return "Summer"
    if m in (6, 7, 8, 9): return "SW Monsoon"
    if m in (10, 11, 12): return "NE Monsoon"
    return "Winter"


df["season"] = df["month"].map(season)
df.to_csv(os.path.join(OUT, "weather_clean.csv"), index=False)


# ---------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------
print("\n=== SUMMARY STATISTICS ===")
print(df[num_cols].describe().round(2))

# 4.1 Temperature trend (daily + 30 day rolling)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df["date"], df["temp_avg"], alpha=.25, color="tab:orange", label="Daily avg")
ax.plot(df["date"], df["temp_avg"].rolling(30).mean(), color="firebrick", lw=2, label="30-day mean")
ax.set(title="Temperature Trend (2020-2025)", xlabel="Date", ylabel="Temp (°C)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/01_temperature_trend.png", dpi=150); plt.close()

# 4.2 Monthly rainfall
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = df.groupby("month_name").agg(rain=("rainfall", "sum"), temp=("temp_avg", "mean"),
                                       hum=("humidity", "mean")).reindex(month_order)
monthly["rain"] /= df["year"].nunique()  # avg per year
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(monthly.index, monthly["rain"], color="steelblue")
ax1.set_ylabel("Avg rainfall per year (mm)")
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["temp"], color="crimson", marker="o", lw=2)
ax2.set_ylabel("Avg temp (°C)"); ax2.grid(False)
ax1.set_title("Monthly Rainfall vs Temperature")
plt.tight_layout(); plt.savefig(f"{OUT}/02_monthly_rain_temp.png", dpi=150); plt.close()

# 4.3 Seasonal boxplot
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="season", y="temp_avg", order=["Winter","Summer","SW Monsoon","NE Monsoon"],
            hue="season", palette="coolwarm", legend=False, ax=ax)
ax.set_title("Temperature Distribution by Season")
plt.tight_layout(); plt.savefig(f"{OUT}/03_seasonal_boxplot.png", dpi=150); plt.close()

# 4.4 Heatmap month x year (avg temp)
pivot = df.pivot_table(index="year", columns="month", values="temp_avg", aggfunc="mean")
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
            xticklabels=month_order)
ax.set_title("Average Temperature Heatmap (Year x Month)")
plt.tight_layout(); plt.savefig(f"{OUT}/04_temp_heatmap.png", dpi=150); plt.close()

# 4.5 Yearly rainfall
yearly = df.groupby("year").agg(rain=("rainfall", "sum"), rainy_days=("is_rainy", "sum"),
                                temp=("temp_avg", "mean"))
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=yearly.index, y=yearly["rain"], color="teal", ax=ax)
for i, v in enumerate(yearly["rain"]):
    ax.text(i, v + 5, f"{v:.0f}", ha="center")
ax.set(title="Total Annual Rainfall", ylabel="mm")
plt.tight_layout(); plt.savefig(f"{OUT}/05_yearly_rainfall.png", dpi=150); plt.close()


# ---------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ---------------------------------------------------------------
corr = df[num_cols + ["temp_avg"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/06_correlation.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df.sample(600, random_state=1), x="humidity", y="temp_max",
                hue="season", alpha=.7, ax=ax)
ax.set_title("Humidity vs Max Temperature")
plt.tight_layout(); plt.savefig(f"{OUT}/07_humidity_vs_temp.png", dpi=150); plt.close()


# ---------------------------------------------------------------
# 6. LONG-TERM TREND (Linear Regression)
# ---------------------------------------------------------------
df["t"] = (df["date"] - df["date"].min()).dt.days
lr = LinearRegression().fit(df[["t"]], df["temp_avg"])
per_year = lr.coef_[0] * 365
print(f"\nTemperature trend: {per_year:+.3f} °C per year")


# ---------------------------------------------------------------
# 7. EXTREME EVENTS
# ---------------------------------------------------------------
hot_thr = df["temp_max"].quantile(0.95)
heavy_rain_thr = 50
hot_days = df[df["temp_max"] >= hot_thr]
heavy = df[df["rainfall"] >= heavy_rain_thr]
print(f"Heatwave threshold (95th pct): {hot_thr:.1f}°C -> {len(hot_days)} days")
print(f"Heavy rain days (>= {heavy_rain_thr} mm): {len(heavy)}")

fig, ax = plt.subplots(figsize=(8, 5))
hot_days.groupby("year").size().plot(kind="bar", color="orangered", ax=ax)
ax.set(title="Hot Days (Top 5% temp) per Year", ylabel="Days")
plt.tight_layout(); plt.savefig(f"{OUT}/08_hot_days.png", dpi=150); plt.close()


# ---------------------------------------------------------------
# 8. RAIN PREDICTION MODEL (will it rain tomorrow?)
# ---------------------------------------------------------------
ml = df.copy()
ml["rain_tomorrow"] = ml["is_rainy"].shift(-1)
ml["rain_yesterday"] = ml["is_rainy"].shift(1)
ml = ml.dropna()
features = ["temp_max", "temp_min", "humidity", "wind_speed", "pressure",
            "month", "rain_yesterday", "is_rainy"]
X, y = ml[features], ml["rain_tomorrow"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_tr, y_tr)
pred = rf.predict(X_te)
acc = accuracy_score(y_te, pred)
print(f"\nRain prediction accuracy: {acc:.2%}")
print(classification_report(y_te, pred, target_names=["No Rain", "Rain"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
axes[0].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
imp.plot(kind="barh", color="seagreen", ax=axes[1]); axes[1].set_title("Feature Importance")
plt.tight_layout(); plt.savefig(f"{OUT}/09_rain_model.png", dpi=150); plt.close()


# ---------------------------------------------------------------
# 9. AUTO-GENERATED INSIGHTS
# ---------------------------------------------------------------
hottest = monthly["temp"].idxmax(); wettest = monthly["rain"].idxmax()
driest = monthly["rain"].idxmin()
wet_year = yearly["rain"].idxmax(); dry_year = yearly["rain"].idxmin()
rain_corr = corr.loc["rainfall", "humidity"]

insights = f"""WEATHER PATTERN ANALYSIS - KEY INSIGHTS
=======================================
1. Dataset: {len(df)} daily records ({df['date'].min().date()} to {df['date'].max().date()}).
2. Hottest month: {hottest} ({monthly.loc[hottest,'temp']:.1f} °C avg).
3. Wettest month: {wettest} ({monthly.loc[wettest,'rain']:.0f} mm/yr avg); driest: {driest}.
4. Wettest year: {wet_year} ({yearly.loc[wet_year,'rain']:.0f} mm); driest year: {dry_year} ({yearly.loc[dry_year,'rain']:.0f} mm).
5. Long-term temperature trend: {per_year:+.3f} °C per year ({"warming" if per_year > 0 else "cooling"}).
6. Rainfall-humidity correlation: {rain_corr:.2f} (higher humidity -> more rain).
7. Hot days (>= {hot_thr:.1f} °C): {len(hot_days)}; heavy rain days (>= 50 mm): {len(heavy)}.
8. Random Forest rain-tomorrow model accuracy: {acc:.1%}.
   Top predictors: {', '.join(imp.index[::-1][:3])}.

RECOMMENDATIONS
- Plan outdoor/agri activities outside the monsoon peaks.
- Monitor rising hot-day counts for heat-stress preparedness.
- Top model features (see chart 09) are the strongest rain signals.
"""
print("\n" + insights)
open(f"{OUT}/insights.txt", "w").write(insights)
print("All charts + cleaned data saved in ./output")
