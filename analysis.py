import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic styling
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Create output directory for images
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------------------
# 🧩 6. Reusable Function (Modular Plotting)
# -------------------------------
def plot_top(series, title, filename):
    """
    Improves modularity and reusability.
    Saves a bar plot of top categories.
    """
    plt.figure(figsize=(10, 5))
    top = series.value_counts().head(10)
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.title(title, weight='bold')
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.close()

print("🚀 Starting Advanced Data Analysis...")

# Load dataset
try:
    df = pd.read_csv('netflix.csv')
except FileNotFoundError:
    print("❌ Error: netflix.csv not found!")
    exit()

# Data Cleaning
df = df.dropna(subset=['type', 'country', 'release_year', 'rating'])

# -------------------------------
# 🚀 1. Content Added Over Time (Platform Growth)
# -------------------------------
print("📊 Analyzing Platform Growth (Date Added)...")
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['year_added'] = df['date_added'].dt.year

# Filter for recent years for better trend visibility
added_trend = df['year_added'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
added_trend.plot(color='red', marker='o', linewidth=2)
plt.title("Netflix Content Library Growth (Titles Added Per Year)", weight='bold')
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{output_dir}/platform_growth.png")
plt.close()

# -------------------------------
# 🌍 2. Country-wise Comparison (Top 5 Focus)
# -------------------------------
print("📊 Comparing Top 5 Countries Over Time...")
top5 = df['country'].value_counts().head(5).index
filtered = df[df['country'].isin(top5)]
# Focus on content released since 2010
country_year = filtered[filtered['release_year'] >= 2010].groupby(['release_year', 'country']).size().unstack()

country_year.plot(figsize=(10, 5), marker='s', markersize=4)
plt.title("Production Trend: Top 5 Content Hubs (2010-Present)", weight='bold')
plt.xlabel("Year of Release")
plt.ylabel("Titles Produced")
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/country_comparison.png")
plt.close()

# -------------------------------
# 🎯 3. Duration Analysis (Movie Runtime)
# -------------------------------
print("📊 Analyzing Movie Durations...")
df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
movies = df[df['type'] == 'Movie']

plt.figure(figsize=(10, 5))
sns.histplot(movies['duration_num'].dropna(), bins=30, kde=True, color='red')
plt.title("Distribution of Movie Runtimes", weight='bold')
plt.xlabel("Duration (Minutes)")
plt.ylabel("Frequency")
plt.savefig(f"{output_dir}/duration_distribution.png")
plt.close()

# -------------------------------
# 🧠 4. Content Rating Analysis
# -------------------------------
print("📊 Analyzing Audience Ratings...")
plot_top(df['rating'], "Top 10 Content Ratings Distribution", "rating_distribution")

# -------------------------------
# 🔥 5. Heatmap (Type vs Rating)
# -------------------------------
print("📊 Generating Type vs Rating Heatmap...")
pivot = df.pivot_table(index='type', columns='rating', aggfunc='size', fill_value=0)
# Select top ratings for a cleaner heatmap
top_ratings = df['rating'].value_counts().head(8).index
pivot_filtered = pivot[top_ratings]

plt.figure(figsize=(12, 5))
sns.heatmap(pivot_filtered, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Count'})
plt.title("Content Type vs Audience Rating", weight='bold')
plt.savefig(f"{output_dir}/type_rating_heatmap.png")
plt.close()

# -------------------------------
# 📊 Simple EDA (Existing Features with Upgraded Functions)
# -------------------------------
print("📊 Updating Standard Visualizations...")
# Content Type Pie Chart
plt.figure(figsize=(7, 7))
type_counts = df['type'].value_counts()
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=['#E50914', '#221F1F'], explode=(0.05, 0))
plt.title("Movies vs TV Shows Ratio", weight='bold')
plt.savefig(f"{output_dir}/distribution_pie.png")
plt.close()

# Top Genres
genres = df['listed_in'].str.split(', ', expand=True).stack()
plot_top(genres, "Top 10 Genres on Netflix", "top_genres")

print("\n" + "="*50)
print("✨ Advanced Analysis Complete!")
print(f"📂 All visualizations saved to the '{output_dir}/' folder.")
print("="*50)
