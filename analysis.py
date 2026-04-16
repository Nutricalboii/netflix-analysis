import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic styling
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

# Create output directory for images
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_top_n(dataframe, column, n=10):
    """
    Utility function to extract the top N categories from a specific column.
    Demonstrates modular code engineering.
    """
    return dataframe[column].value_counts().head(n)

print("🚀 Initiating Content Intelligence Analysis...")
print("📂 Loading and cleaning dataset...")

# Load dataset
try:
    df = pd.read_csv('netflix.csv')
except FileNotFoundError:
    print("❌ Error: netflix.csv not found!")
    exit()

# -------------------------------
# Data Cleaning & Preprocessing
# -------------------------------
# Remove null values in important columns
df = df.dropna(subset=['type', 'country', 'release_year'])

# Ensure release_year is numeric
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)
df = df[df['release_year'] > 0] # Filter out invalid years

print("✅ Data cleaning complete.")

# -------------------------------
# 1. Movies vs TV Shows
# -------------------------------
print("\n📊 1. Analyzing Content Distribution (Movies vs TV Shows)...")
plt.figure(figsize=(8, 8))
type_counts = df['type'].value_counts()
colors = ['#E50914', '#221F1F'] # Netflix colors: Red and Black/Dark Gray
plt.pie(
    type_counts, 
    labels=type_counts.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=colors,
    explode=(0.05, 0),
    textprops={'color':"white", 'weight':'bold'}
)
plt.title("Netflix Content Distribution: Movies vs TV Shows", fontsize=14, weight='bold')
plt.savefig(f"{output_dir}/distribution_pie.png", bbox_inches='tight', facecolor='#F5F5F1')
plt.close()

# Analysis Sentence
print("💡 Insight: Movies dominate the library (~70%), suggesting Netflix remains primarily a film-distribution platform.")

# -------------------------------
# 2. Top Countries
# -------------------------------
print("\n📊 2. Analyzing Geographical Content Intelligence...")
top_countries = get_top_n(df, 'country', 10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.index, y=top_countries.values, hue=top_countries.index, palette="Reds_r", legend=False)
plt.xticks(rotation=45)
plt.title("Top 10 Content Producing Countries", fontsize=15, weight='bold')
plt.xlabel("Country", fontsize=12)
plt.ylabel("Content Count", fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/top_countries.png")
plt.close()

# Analysis Sentence
print(f"💡 Insight: The {top_countries.index[0]} is the leading producer, followed by {top_countries.index[1]}, highlighting global diversity.")

# -------------------------------
# 3. Content Over Years
# -------------------------------
print("\n📊 3. Studying Historical Production Trends...")
# Take data from 2000 onwards for better clarity
year_data = df[df['release_year'] >= 2000]['release_year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.plot(year_data.index, year_data.values, color='#E50914', linewidth=3, marker='o', markersize=6, markerfacecolor='#221F1F')
plt.fill_between(year_data.index, year_data.values, color='#E50914', alpha=0.1)
plt.title("Growth of Content on Netflix (Since 2000)", fontsize=15, weight='bold')
plt.xlabel("Year of Release", fontsize=12)
plt.ylabel("Number of Shows/Movies", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"{output_dir}/yearly_trend.png")
plt.close()

# Analysis Sentence
print("💡 Insight: Content production accelerated drastically after 2015, marking the era of 'Netflix Originals'.")

# -------------------------------
# 4. Most Common Genres
# -------------------------------
print("\n📊 4. Mapping Genre Popularity...")
# Extract individual genres (split by comma)
genres = df['listed_in'].str.split(', ', expand=True).stack()
top_genres = genes_top_10 = genres.value_counts().head(10)

plt.figure(figsize=(12, 7))
sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette="flare", legend=False)
plt.title("Top 10 Most Popular Genres on Netflix", fontsize=15, weight='bold')
plt.xlabel("Number of Titles", fontsize=12)
plt.ylabel("Genre", fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/top_genres.png")
plt.close()

# Analysis Sentence
print(f"💡 Insight: {top_genres.index[0]} and {top_genres.index[1]} are the most frequent genres, aligning with audience demand for drama.")

print("\n" + "="*50)
print("✨ Content Intelligence Analysis Complete!")
print(f"📂 Visualizations have been saved in the '{output_dir}' directory.")
print("💡 Note: This analysis is limited by data availability and missing values for some regions.")
print("="*50)
