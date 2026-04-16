import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# -------------------------------
# 🎨 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="NCIP | Netflix Content Intelligence Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #141414;
        color: white;
    }
    .stMetric {
        background-color: #221F1F;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E50914;
    }
    div[data-testid="stSidebar"] {
        background-color: #000000;
    }
    h1, h2, h3 {
        color: #E50914 !important;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-med { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 📂 Data Intelligence Engine
# -------------------------------
@st.cache_data
def load_data():
    df_raw = pd.read_csv('netflix.csv')
    total_raw = len(df_raw)
    clean_raw = df_raw.dropna(subset=['type', 'country', 'release_year', 'rating']).shape[0]
    reliability_index = clean_raw / total_raw
    
    df = df_raw.copy()
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['content_age'] = df['year_added'] - df['release_year']
    
    def calculate_score(row):
        score = 0
        if row['type'] == 'Movie': score += 1
        if row['rating'] == 'TV-MA': score += 2
        if row['release_year'] > 2015: score += 2
        return score
    
    df['content_score'] = df.apply(calculate_score, axis=1)
    df = df.dropna(subset=['type', 'country', 'release_year', 'rating', 'description'])
    return df, reliability_index

df_raw, data_reliability = load_data()

# -------------------------------
# 🤖 AI Engine: Recommender
# -------------------------------
@st.cache_resource
def get_recommender(data):
    data = data.copy()
    data['metadata_soup'] = data['type'] + " " + data['listed_in'] + " " + data['rating'] + " " + data['description']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['metadata_soup'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = get_recommender(df_raw)
indices = pd.Series(df_raw.index, index=df_raw['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, df=df_raw):
    if title not in indices: return pd.DataFrame(), []
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'type', 'listed_in', 'description']], [i[1] for i in sim_scores]

# -------------------------------
# 🧬 AI Engine: Market Segmentation
# -------------------------------
@st.cache_data
def get_market_segments(data):
    country_stats = data.groupby('country').agg({'title': 'count', 'content_score': 'mean', 'year_added': 'mean'}).reset_index()
    tv_ratio = data[data['type'] == 'TV Show'].groupby('country').size() / data.groupby('country').size()
    country_stats['tv_ratio'] = country_stats['country'].map(tv_ratio).fillna(0)
    country_stats.columns = ['country', 'total_titles', 'avg_score', 'avg_recency', 'tv_ratio']
    
    features = ['total_titles', 'avg_score', 'tv_ratio', 'avg_recency']
    scaled_features = StandardScaler().fit_transform(country_stats[features])
    country_stats['cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled_features)
    
    def label_archetype(row):
        if row['total_titles'] > 100: return "Global Powerhouse"
        if row['tv_ratio'] > 0.5: return "Binge-Culture Hub"
        if row['avg_score'] > 3.0: return "Strategic High-Value Hub"
        return "Emerging Specialist"
    
    country_stats['archetype'] = country_stats.apply(label_archetype, axis=1)
    return country_stats

market_df = get_market_segments(df_raw)

# -------------------------------
# 🔮 AI Engine: Growth Forecasting
# -------------------------------
def get_growth_forecast(filtered_df):
    history = filtered_df['year_added'].value_counts().sort_index().reset_index()
    history.columns = ['Year', 'Count']
    history = history[history['Year'] >= 2008] # Focus on modern growth
    
    if len(history) < 3: return None, None # Not enough data for trend
    
    X = history[['Year']].values
    y = history['Count'].values
    
    model = LinearRegression().fit(X, y)
    
    # Predict 2024, 2025, 2026
    future_years = np.array([[2024], [2025], [2026], [2027]])
    future_preds = model.predict(future_years)
    
    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Count': future_preds.flatten()})
    forecast_df['Type'] = 'Forecast'
    history['Type'] = 'Historical'
    
    return pd.concat([history, forecast_df]), model.coef_[0]

# -------------------------------
# 🧠 Strategic Insight Engine
# -------------------------------
def generate_insights(filtered_df, total_df):
    insights = []
    if filtered_df.empty: return [("No data available", "Low")]
    density = len(filtered_df) / len(total_df)
    def get_conf(val=density):
        if val > 0.1: return "High"
        if val > 0.02: return "Medium"
        return "Low"
    counts = filtered_df['type'].value_counts()
    if counts.get('Movie', 0) > counts.get('TV Show', 0): insights.append(("Movies dominate the platform share.", get_conf()))
    else: insights.append(("TV Shows lead this segment (Binge-culture focus).", get_conf()))
    if not filtered_df['year_added'].empty:
        max_year = filtered_df['year_added'].value_counts().idxmax()
        if max_year > 2015: insights.append(("Strategic pivot to 'Originals' detected post-2015.", get_conf()))
    top_rating = filtered_df['rating'].value_counts().idxmax()
    if top_rating in ['TV-MA', 'R']: insights.append((f"Adult audience ({top_rating}) is the primary target.", get_conf()))
    if filtered_df['content_score'].mean() > 3.0: insights.append(("Strategic Content Score is peaking in this segment.", "High"))
    return insights

# -------------------------------
# 🛠️ Sidebar Filters
# -------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
st.sidebar.title("Intelligence Filters")
content_types = st.sidebar.multiselect("Select Content Type", options=df_raw['type'].unique(), default=df_raw['type'].unique())
min_year, max_year = int(df_raw['release_year'].min()), int(df_raw['release_year'].max())
year_range = st.sidebar.slider("Release Year Range", min_value=min_year, max_value=max_year, value=(2010, max_year))
selected_countries = st.sidebar.multiselect("Select Countries", options=sorted(df_raw['country'].unique()), default=["United States", "India", "United Kingdom", "South Korea", "Japan"])

df = df_raw[(df_raw['type'].isin(content_types)) & (df_raw['release_year'].between(year_range[0], year_range[1])) & (df_raw['country'].isin(selected_countries))]

# -------------------------------
# 📊 Dashboard Header
# -------------------------------
st.title("🎬 Netflix Content Intelligence Platform (NCIP)")
st.markdown("Transforming raw streaming data into actionable strategic insights.")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Titles", len(df))
col2.metric("Movies", len(df[df['type'] == 'Movie']))
col3.metric("TV Shows", len(df[df['type'] == 'TV Show']))
col4.metric("Avg Strat Score", f"{df['content_score'].mean():.2f}")
col5.metric("Data Reliability", f"{data_reliability:.2%}")
st.divider()

st.subheader("💡 Strategic Insights (Prescriptive)")
insights = generate_insights(df, df_raw)
cols = st.columns(len(insights) if insights else 1)
for i, (text, conf) in enumerate(insights):
    conf_class = "confidence-high" if conf == "High" else "confidence-med" if conf == "Medium" else "confidence-low"
    cols[i % len(cols)].info(f"**{text}**\n\nConfidence: <span class='{conf_class}'>{conf}</span>", icon="ℹ️")
st.divider()

# Main UI Tabs
tab_core, tab_advanced, tab_prescriptive, tab_ai, tab_seg = st.tabs(["📊 Core Intelligence", "🧠 Behavioral Analysis", "🎯 Strategic Opportunities", "🤖 AI Recommendations", "🧬 Market Segmentation"])

with tab_core:
    c1, c2 = st.columns(2)
    with c1:
        added_trend = df['year_added'].value_counts().sort_index().reset_index()
        added_trend.columns = ['Year', 'Count']
        st.plotly_chart(px.line(added_trend, x='Year', y='Count', title="Content Added over Time", line_shape='spline', color_discrete_sequence=['#E50914']).update_layout(template="plotly_dark"), use_container_width=True)
    with c2:
        top_countries_df = df['country'].value_counts().head(10).reset_index(); top_countries_df.columns = ['Country', 'Count']
        st.plotly_chart(px.bar(top_countries_df, x='Count', y='Country', orientation='h', title="Top Production Hubs", color='Count', color_continuous_scale='Reds').update_layout(template="plotly_dark"), use_container_width=True)

with tab_advanced:
    st.subheader("🧠 Behavioral & Structural Intelligence")
    ac1, ac2 = st.columns(2)
    with ac1:
        expansion = df.groupby('year_added')['country'].nunique().reset_index(); expansion.columns = ['Year', 'Unique Countries']
        st.plotly_chart(px.area(expansion, x='Year', y='Unique Countries', title="Geographical Expansion", color_discrete_sequence=['#E50914']).update_layout(template="plotly_dark"), use_container_width=True)
    with ac2:
        mix = df.groupby(['year_added', 'type']).size().unstack().fillna(0).reset_index()
        st.plotly_chart(px.bar(mix, x='year_added', y=['Movie', 'TV Show'], title="Content Mix Evolution", color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'}).update_layout(template="plotly_dark"), use_container_width=True)

with tab_prescriptive:
    st.subheader("🎯 Strategic Opportunities & Decision Support")
    pc1, pc2 = st.columns(2)
    with pc1:
        country_growth = df.groupby('country')['year_added'].mean().sort_values(ascending=False).head(10).reset_index(); country_growth.columns = ['Country', 'Avg Year Added']
        st.plotly_chart(px.bar(country_growth, x='Avg Year Added', y='Country', orientation='h', title="Fastest Growing Sourcing Countries", color='Avg Year Added', color_continuous_scale='Reds', range_x=[min(max_year-5, 2015), max_year]).update_layout(template="plotly_dark"), use_container_width=True)
    with pc2:
        segment = df.groupby(['type', 'rating'])['content_score'].mean().sort_values(ascending=False).head(8).reset_index()
        st.plotly_chart(px.bar(segment, x='content_score', y='rating', color='type', title="Avg Strategy Score by Segment", barmode='group', color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'}).update_layout(template="plotly_dark"), use_container_width=True)
    
    st.divider()
    
    # 🔮 Predictive Growth Forecast
    st.subheader("🔮 Predictive Growth Forecast")
    st.markdown("Forecasting catalog expansion trends for the next 3 years based on historical growth slopes.")
    f_col1, f_col2 = st.columns([2, 1])
    forecast_df, slope = get_growth_forecast(df)
    
    if forecast_df is not None:
        with f_col1:
            fig_fc = px.line(forecast_df, x='Year', y='Count', color='Type', title="AI-Projected Catalog Growth (Linear Regression)",
                             line_dash='Type', color_discrete_map={'Historical': '#E50914', 'Forecast': '#ffffff'})
            fig_fc.update_layout(template="plotly_dark")
            st.plotly_chart(fig_fc, use_container_width=True)
        with f_col2:
            st.markdown("### 📋 Strategy Roadmap")
            if slope > 10:
                st.success(f"**Trend: Aggressive Expansion**\n\nThe slope ({slope:.1f}) indicates a high acceleration in content sourcing. Recommendation: Diversify production hubs to mitigate saturation risks.")
            elif slope > 0:
                st.info(f"**Trend: Steady Growth**\n\nThe slope ({slope:.1f}) shows consistent platform scaling. Recommendation: Focus on quality-over-quantity to improve Content Strategy Scores.")
            else:
                st.warning(f"**Trend: Growth Plateau**\n\nNegative slope ({slope:.1f}) detected. Recommendation: Strategic pivot needed. Explore emerging markets or new content formats.")
    else:
        st.warning("Insufficient historical data to generate a reliable growth forecast for this segment.")

with tab_ai:
    st.subheader("🤖 SmartContent AI Recommender")
    target_title = st.selectbox("Search for a Title to see Recommendations", options=sorted(df_raw['title'].unique()))
    if target_title:
        rec_df, scores = get_recommendations(target_title)
        st.markdown(f"### Top 5 Recommendations for: **{target_title}**")
        cols = st.columns(5); [(cols[i].markdown(f"**{row['title']}**"), cols[i].caption(f"Match: {scores[i]:.0%}"), cols[i].write(row['description'][:100] + "...")) for i, (idx, row) in enumerate(rec_df.iterrows())]

with tab_seg:
    st.subheader("🧬 Market Segmentation")
    st.plotly_chart(px.scatter_3d(market_df, x='total_titles', y='tv_ratio', z='avg_score', color='archetype', size='total_titles', hover_name='country', title="Global Market Sourcing Archetypes", labels={'total_titles': 'Volume', 'tv_ratio': 'TV Show Ratio', 'avg_score': 'Strat Score'}, color_discrete_sequence=px.colors.qualitative.Reds).update_layout(template="plotly_dark", scene=dict(bgcolor='#141414')), use_container_width=True)

# -------------------------------
# 🏁 Footer
# -------------------------------
st.divider()
st.markdown("""<div style="text-align: center; color: gray;">NCIP: Netflix Content Intelligence Platform | Prophetic Tier Edition | Built for Strategic Data Excellence</div>""", unsafe_allow_html=True)
