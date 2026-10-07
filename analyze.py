import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
})

CHART_DIR = "/home/claude/sales_project/charts"

df = pd.read_csv("/home/claude/sales_project/data/sales_data.csv", parse_dates=["OrderDate"])
df["Month"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
df["Year"] = df["OrderDate"].dt.year
df["MonthNum"] = df["OrderDate"].dt.month
df["Weekday"] = df["OrderDate"].dt.day_name()

# -----------------------------------------------------------------
# 1. Monthly revenue trend + 3-month moving average
# -----------------------------------------------------------------
monthly = df.groupby("Month")["Revenue"].sum().reset_index()
monthly["MA3"] = monthly["Revenue"].rolling(3).mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly["Month"], monthly["Revenue"] / 1e5, marker="o", color="#4C6FFF",
        linewidth=2, label="Monthly Revenue")
ax.plot(monthly["Month"], monthly["MA3"] / 1e5, linestyle="--", color="#FF6B6B",
        linewidth=2, label="3-Month Moving Avg")
ax.set_title("Monthly Sales Revenue Trend (2024-2025)", fontsize=14, fontweight="bold")
ax.set_ylabel("Revenue (₹ Lakhs)")
ax.set_xlabel("Month")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/01_monthly_trend.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 2. Year-over-year comparison by month
# -----------------------------------------------------------------
pivot = df.groupby(["Year", "MonthNum"])["Revenue"].sum().unstack(0) / 1e5
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 5))
width = 0.35
x = np.arange(12)
for i, year in enumerate(pivot.columns):
    ax.bar(x + (i - 0.5) * width, pivot[year].reindex(range(1, 13)).values, width,
           label=str(year), color=["#4C6FFF", "#FFB84C"][i % 2])
ax.set_xticks(x)
ax.set_xticklabels(month_labels)
ax.set_title("Year-over-Year Monthly Revenue Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Revenue (₹ Lakhs)")
ax.legend(title="Year")
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/02_yoy_comparison.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 3. Revenue share by category
# -----------------------------------------------------------------
cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
colors = ["#4C6FFF", "#6E8CFF", "#93A9FF", "#FFB84C", "#FF8A6B"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(cat_rev.index[::-1], cat_rev.values[::-1] / 1e5, color=colors[::-1])
ax.set_title("Total Revenue by Product Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Revenue (₹ Lakhs)")
for bar in bars:
    w = bar.get_width()
    ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}L", va="center", fontsize=10)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/03_category_revenue.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 4. Regional performance
# -----------------------------------------------------------------
region_rev = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    region_rev.values, labels=region_rev.index, autopct="%1.1f%%",
    colors=["#4C6FFF", "#6E8CFF", "#FFB84C", "#FF8A6B"], startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
ax.set_title("Revenue Share by Region", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/04_region_share.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 5. Heatmap: Region x Month revenue intensity
# -----------------------------------------------------------------
heat = df.groupby(["Region", "Month"])["Revenue"].sum().unstack(0) / 1e5
fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(heat.T.values, aspect="auto", cmap="Blues")
ax.set_yticks(range(len(heat.columns)))
ax.set_yticklabels(heat.columns)
ax.set_xticks(range(len(heat.index)))
ax.set_xticklabels([d.strftime("%b-%y") for d in heat.index], rotation=45, ha="right")
ax.set_title("Revenue Intensity: Region × Month (₹ Lakhs)", fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("₹ Lakhs")
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/05_region_month_heatmap.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 6. Top 10 products
# -----------------------------------------------------------------
top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(top_products.index[::-1], top_products.values[::-1] / 1e5, color="#4C6FFF")
ax.set_title("Top 10 Products by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("Revenue (₹ Lakhs)")
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}L", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/06_top_products.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 7. Day-of-week sales pattern
# -----------------------------------------------------------------
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
wk = df.groupby("Weekday")["Revenue"].sum().reindex(weekday_order) / 1e5
fig, ax = plt.subplots(figsize=(8, 4.5))
colors_wk = ["#4C6FFF"]*5 + ["#FF6B6B"]*2
ax.bar(wk.index, wk.values, color=colors_wk)
ax.set_title("Revenue by Day of Week", fontsize=14, fontweight="bold")
ax.set_ylabel("Revenue (₹ Lakhs)")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/07_weekday_pattern.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# Summary statistics (printed for report text)
# -----------------------------------------------------------------
total_revenue = df["Revenue"].sum()
total_orders = len(df)
avg_order_value = df["Revenue"].mean()
rev_2024 = df[df.Year == 2024]["Revenue"].sum()
rev_2025 = df[df.Year == 2025]["Revenue"].sum()
yoy_growth = (rev_2025 - rev_2024) / rev_2024 * 100
best_month = monthly.loc[monthly["Revenue"].idxmax()]
worst_month = monthly.loc[monthly["Revenue"].idxmin()]
top_category = cat_rev.index[0]
top_region = region_rev.index[0]
top_product = top_products.index[0]
best_weekday = wk.idxmax()

print("=== SUMMARY STATS ===")
print(f"Total Revenue: ₹{total_revenue:,.0f}")
print(f"Total Orders: {total_orders:,}")
print(f"Avg Order Value: ₹{avg_order_value:,.0f}")
print(f"2024 Revenue: ₹{rev_2024:,.0f}")
print(f"2025 Revenue: ₹{rev_2025:,.0f}")
print(f"YoY Growth: {yoy_growth:.1f}%")
print(f"Best Month: {best_month['Month'].strftime('%b %Y')} (₹{best_month['Revenue']:,.0f})")
print(f"Weakest Month: {worst_month['Month'].strftime('%b %Y')} (₹{worst_month['Revenue']:,.0f})")
print(f"Top Category: {top_category} (₹{cat_rev.iloc[0]:,.0f})")
print(f"Top Region: {top_region} ({region_rev.iloc[0]/total_revenue*100:.1f}% share)")
print(f"Top Product: {top_product} (₹{top_products.iloc[0]:,.0f})")
print(f"Best Weekday: {best_weekday}")

# Save summary to a text file for reuse in report writing
with open("/home/claude/sales_project/data/summary_stats.txt", "w") as f:
    f.write(f"Total Revenue: Rs {total_revenue:,.0f}\n")
    f.write(f"Total Orders: {total_orders:,}\n")
    f.write(f"Avg Order Value: Rs {avg_order_value:,.0f}\n")
    f.write(f"2024 Revenue: Rs {rev_2024:,.0f}\n")
    f.write(f"2025 Revenue: Rs {rev_2025:,.0f}\n")
    f.write(f"YoY Growth: {yoy_growth:.1f}%\n")
    f.write(f"Best Month: {best_month['Month'].strftime('%b %Y')} (Rs {best_month['Revenue']:,.0f})\n")
    f.write(f"Weakest Month: {worst_month['Month'].strftime('%b %Y')} (Rs {worst_month['Revenue']:,.0f})\n")
    f.write(f"Top Category: {top_category} (Rs {cat_rev.iloc[0]:,.0f})\n")
    f.write(f"Top Region: {top_region} ({region_rev.iloc[0]/total_revenue*100:.1f}% share)\n")
    f.write(f"Top Product: {top_product} (Rs {top_products.iloc[0]:,.0f})\n")
    f.write(f"Best Weekday: {best_weekday}\n")

print("\nAll charts saved to", CHART_DIR)
