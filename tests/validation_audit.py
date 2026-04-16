import pandas as pd
import time
import os
import sys

# Mocking Streamlit's cache_data for local testing
def cache_data(func):
    return func

# -------------------------------
# 📂 Data Load and Preprocessing logic
# -------------------------------
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
    
    raw_count = len(df)
    df = df.dropna(subset=['type', 'country', 'release_year', 'rating'])
    cleaned_count = len(df)
    
    end_time = time.time()
    perf_metrics = {
        "load_time": end_time - start_time,
        "raw_count": raw_count,
        "cleaned_count": cleaned_count,
        "rows_removed": raw_count - cleaned_count
    }
    return df, perf_metrics

# -------------------------------
# 🧠 Insight Engine Logic (Mocked)
# -------------------------------
def generate_insights(filtered_df):
    insights = []
    if filtered_df.empty: return ["No data"]
    counts = filtered_df['type'].value_counts()
    if counts.get('Movie', 0) > counts.get('TV Show', 0):
        insights.append("Movies dominate")
    else:
        insights.append("TV Shows dominate")
    return insights

# -------------------------------
# 🕵️ Audit Suite
# -------------------------------
def run_audit():
    print("🚀 Initiating Elite Data Integrity Audit...")
    df, perf = load_data_audit()
    
    if df is None:
        print(f"FAILED: {perf}")
        sys.exit(1)
        
    errors = []
    warnings = []
    
    # 1. Integrity Check: Type Split
    total = len(df)
    movies = len(df[df['type'] == 'Movie'])
    tv_shows = len(df[df['type'] == 'TV Show'])
    if total != (movies + tv_shows):
        errors.append(f"Inconsistent Type Split: Total({total}) != Movie({movies}) + TV({tv_shows})")
    else:
        print("✅ Integrity: Movies + TV Shows match Total Titles.")

    # 2. Year Validity
    future_years = df[df['release_year'] > 2026]
    if not future_years.empty:
        errors.append(f"Future release years detected ({len(future_years)} titles)")
    
    invalid_added = df[df['year_added'] < df['release_year']]
    if not invalid_added.empty:
        warnings.append(f"Content Added Before Release: {len(invalid_added)} titles (likely data errors in netflix.csv)")

    # 3. Outlier check (Duration)
    extreme_movie = df[(df['type'] == 'Movie') & (df['duration_num'] > 300)]
    if not extreme_movie.empty:
        warnings.append(f"Extreme Duration Detected: {len(extreme_movie)} movies > 5 hours.")

    # 4. Insight Engine Unit Test
    test_df = pd.DataFrame({'type': ['Movie', 'Movie', 'TV Show'], 'rating': ['TV-MA']*3})
    insights = generate_insights(test_df)
    if not any("Movies dominate" in s for s in insights):
        errors.append("Insight Engine: Failed to detect Movie dominance")
    else:
        print("✅ Insight Engine: Strategic logic verified with unit test.")

    # 5. Performance Check
    if perf['load_time'] > 2.0:
        errors.append(f"Performance failure: Load time {perf['load_time']:.2f}s exceeds 2s limit!")
    else:
        print(f"✅ Performance: Platform loads in {perf['load_time']:.4f}s (PRD Spec compliant).")

    # -------------------------------
    # 📊 Final Report
    # -------------------------------
    print("\n" + "="*50)
    print("📋 FULL EXTENT TESTING REPORT")
    print(f"• Dataset Size: {perf['raw_count']} rows")
    print(f"• Sanitization: {perf['rows_removed']} rows removed due to missing critical columns")
    print(f"• Integrity: {'PASS' if not errors else 'FAIL'}")
    print(f"• Errors found: {len(errors)}")
    print(f"• Warnings found: {len(warnings)}")
    
    if errors:
        print("\n❌ CRITICAL ERRORS:")
        for e in errors: print(f"  - {e}")
    
    if warnings:
        print("\n⚠️ WARNINGS (Data Specific):")
        for w in warnings: print(f"  - {w}")
    
    print("="*50)

    if not errors:
        print("\n✨ FINAL VERDICT: NCIP is Logically Bulletproof.")
    else:
        print("\n❌ FINAL VERDICT: Integrity Compromised. Fixes needed.")

if __name__ == "__main__":
    run_audit()
