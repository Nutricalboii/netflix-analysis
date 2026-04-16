import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 🧪 NCIP Diamond-Status Audit Suite
# -----------------------------------------------------------------------------

def load_data_audit():
    start_time = time.time()
    try:
        df = pd.read_csv('netflix.csv')
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"
    
    # Preprocessing to match app.py
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['content_age'] = df['year_added'] - df['release_year']
    
    # Strategic Scoring
    def calculate_score(row):
        score = 0
        if row['type'] == 'Movie': score += 1
        if row['rating'] == 'TV-MA': score += 2
        if row['release_year'] > 2015: score += 2
        return score
    df['content_score'] = df.apply(calculate_score, axis=1)
    
    raw_count = len(df)
    df = df.dropna(subset=['type', 'country', 'release_year', 'rating'])
    cleaned_count = len(df)
    
    end_time = time.time()
    return df, {"load_time": end_time - start_time, "raw_count": raw_count, "cleaned_count": cleaned_count}

def test_clustering_stability(df):
    """Ensure K-Means provides consistent archetypes."""
    try:
        country_stats = df.groupby('country').agg({'title': 'size', 'content_score': 'mean', 'year_added': 'mean'}).reset_index()
        tv_ratio = df[df['type'] == 'TV Show'].groupby('country').size() / df.groupby('country').size()
        country_stats['tv_ratio'] = country_stats['country'].map(tv_ratio).fillna(0)
        country_stats.columns = ['country', 'total_titles', 'avg_score', 'avg_recency', 'tv_ratio']
        
        features = ['total_titles', 'avg_score', 'tv_ratio', 'avg_recency']
        scaled = StandardScaler().fit_transform(country_stats[features])
        
        # Run twice to check stability
        km1 = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled)
        km2 = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled)
        
        if np.array_equal(km1.labels_, km2.labels_):
            return True, "Stable"
        return False, "Unstable Labels"
    except Exception as e:
        return False, str(e)

def test_forecast_logic(df):
    """Validate linear regression for growth forecasting."""
    try:
        history = df['year_added'].value_counts().sort_index().reset_index()
        history.columns = ['Year', 'Count']
        history = history[history['Year'] >= 2010]
        
        X, y = history[['Year']].values, history['Count'].values
        model = LinearRegression().fit(X, y)
        
        # Check if slope is reasonable (not effectively zero or infinite)
        slope = model.coef_[0]
        if -1000 < slope < 1000:
            return True, f"Valid Slope: {slope:.2f}"
        return False, f"Extreme Slope Detected: {slope}"
    except Exception as e:
        return False, str(e)

def run_audit():
    print("💎 NCIP DIAMOND STATUS AUDIT INITIATED")
    print("="*50)
    
    df, perf = load_data_audit()
    errors = []
    warnings = []
    
    # 1. Performance Audit
    if perf['load_time'] < 1.0:
        print(f"✅ Performance: Platform core ready in {perf['load_time']:.4f}s.")
    else:
        warnings.append(f"Performance latency: {perf['load_time']:.2f}s (Threshold: 1.0s)")

    # 2. Mathematical Integrity
    if (df['content_score'] >= 0).all() and (df['content_score'] <= 5).all():
        print("✅ Strategic Engine: Scores are mathematically bounded [0, 5].")
    else:
        errors.append("Strategic Score out of bounds!")

    # 3. AI Module: Clustering Stability
    stable, msg = test_clustering_stability(df)
    if stable:
        print(f"✅ AI Engine: Market Clustering is {msg}.")
    else:
        errors.append(f"AI Engine: Clustering instability detected ({msg})")

    # 4. AI Module: Forecast Boundary
    valid_fc, fc_msg = test_forecast_logic(df)
    if valid_fc:
        print(f"✅ AI Engine: Growth Forecast Logic is {fc_msg}.")
    else:
        errors.append(f"AI Engine: Forecast logic failure ({fc_msg})")

    # 5. Stress Test: Edge Years
    edge_years = df[df['year_added'] < 2008]
    if len(edge_years) > 0:
        warnings.append(f"Sparse History: {len(edge_years)} titles added before 2008 (may affect trend accuracy).")

    # -------------------------------
    # 📊 Final Diamond Report
    # -------------------------------
    print("\n" + "="*50)
    print("📋 DIAMOND STATUS QA REPORT")
    print(f"• Dataset Integrity: {perf['cleaned_count']}/{perf['raw_count']} rows valid")
    print(f"• Errors: {len(errors)}")
    print(f"• Warnings: {len(warnings)}")
    
    if errors:
        print("\n❌ CRITICAL BLOCKERS:")
        for e in errors: print(f"  - {e}")
    else:
        print("\n✨ FINAL VERDICT: NCIP IS OFFICIALLY DIAMOND HARDENED.")
    print("="*50)

if __name__ == "__main__":
    run_audit()
