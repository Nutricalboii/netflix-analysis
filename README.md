# 🎬 Netflix Data Intelligence Analysis

An advanced data analysis project exploring Netflix's global catalog. This project shifts from basic tracking to deep behavioral insights, covering platform growth, geographical dominance, and audience targeting.

## 📊 Key Insight Visualizations

### 1. Platform Expansion (Temporal Growth)
Analysis of when content was added to the platform, revealing Netflix's aggressive scaling strategy since 2015.
![Platform Growth](plots/platform_growth.png)

### 2. Global Production Trends (Top 5 Hubs)
A comparative trend analysis of the top 5 content-producing countries.
![Country Comparison](plots/country_comparison.png)

### 3. Movie Runtime Distribution
Structural analysis of movie durations, identifying the standard length of Netflix's film catalog.
![Duration Distribution](plots/duration_distribution.png)

### 4. Revenue & Audience Targeting (Heatmap)
A sophisticated mapping of content types against audience ratings (e.g., TV-MA, TV-14), showing Netflix's focus on mature audiences.
![Type Rating Heatmap](plots/type_rating_heatmap.png)

### 5. Content Ratings Distribution
Identifying the primary audience demographics through content ratings.
![Rating Distribution](plots/rating_distribution.png)

### 6. Standard EDA
Overview of content distribution (Movies vs TV Shows) and popular genres.
![Content Distribution](plots/distribution_pie.png)
![Top Genres](plots/top_genres.png)

## 🛠️ Analysis Framework
- **Python**: Core engine for data processing.
- **Pandas**: Advanced preprocessing and categorical data extraction.
- **Matplotlib/Seaborn**: High-resolution visualization with custom stylistic themes.
- **Modular Code**: Implemented reusable plotting functions for scalability.

## 📁 Project Structure
```text
netflix-analysis/
├── netflix.csv           # Raw dataset
├── analysis.py           # Advanced analysis script
├── analysis.ipynb        # Interactive Notebook
└── plots/                # High-resolution PNG exports
```

## 🚀 Getting Started
1. Install dependencies: `pip install pandas matplotlib seaborn`
2. Execute the intelligence engine: `python analysis.py`
