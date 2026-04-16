import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    # Reliability Metric
    total_raw = len(df_raw)
    clean_raw = df_raw.dropna(subset=['type', 'country', 'release_year', 'rating']).shape[0]
    reliability_index = clean_raw / total_raw
    
    # Preprocessing
    df = df_raw.copy()
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['content_age'] = df['year_added'] - df['release_year']
    
    # 🎯 1. Content Strategy Scoring
    def calculate_score(row):
        score = 0
        if row['type'] == 'Movie': score += 1
        if row['rating'] == 'TV-MA': score += 2
        if row['release_year'] > 2015: score += 2
        return score
    
    df['content_score'] = df.apply(calculate_score, axis=1)
    
    df = df.dropna(subset=['type', 'country', 'release_year', 'rating'])
    return df, reliability_index

df_raw, data_reliability = load_data()

# -------------------------------
# 🧠 Strategic Insight Engine (with Confidence)
# -------------------------------
def generate_insights(filtered_df, total_df):
    insights = []
    
    if filtered_df.empty:
        return [("No data available", "Low")]
    
    # Confidence Proxy (Data Density)
    density = len(filtered_df) / len(total_df)
    def get_conf(val=density):
        if val > 0.1: return "High"
        if val > 0.02: return "Medium"
        return "Low"

    # Type Dominance
    counts = filtered_df['type'].value_counts()
    if counts.get('Movie', 0) > counts.get('TV Show', 0):
        insights.append(("Movies dominate the platform share.", get_conf()))
    else:
        insights.append(("TV Shows lead this segment (Binge-culture focus).", get_conf()))
        
    # Growth Spikes
    if not filtered_df['year_added'].empty:
        max_year = filtered_df['year_added'].value_counts().idxmax()
        if max_year > 2015:
            insights.append(("Strategic pivot to 'Originals' detected post-2015.", get_conf()))

    # Audience Focus
    top_rating = filtered_df['rating'].value_counts().idxmax()
    if top_rating in ['TV-MA', 'R']:
        insights.append((f"Adult audience ({top_rating}) is the primary target.", get_conf()))

    # Strategic Table Insight
    avg_score = filtered_df['content_score'].mean()
    if avg_score > 3.0:
        insights.append(("Strategic Content Score is peaking in this segment.", "High"))

    return insights

# -------------------------------
# 🛠️ Sidebar Filters
# -------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
st.sidebar.title("Intelligence Filters")

# Filter: Content Type
content_types = st.sidebar.multiselect(
    "Select Content Type",
    options=df_raw['type'].unique(),
    default=df_raw['type'].unique()
)

# Filter: Year Range
min_year = int(df_raw['release_year'].min())
max_year = int(df_raw['release_year'].max())
year_range = st.sidebar.slider(
    "Release Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(2010, max_year)
)

# Filter: Country
countries = sorted(df_raw['country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=["United States", "India", "United Kingdom", "South Korea", "Japan"]
)

# Apply Filters
df = df_raw[
    (df_raw['type'].isin(content_types)) &
    (df_raw['release_year'].between(year_range[0], year_range[1])) &
    (df_raw['country'].isin(selected_countries))
]

# -------------------------------
# 📊 Dashboard Header
# -------------------------------
st.title("🎬 Netflix Content Intelligence Platform (NCIP)")
st.markdown("Transforming raw streaming data into actionable strategic insights.")

# Key Performance Indicators (KPIs)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Titles", len(df))
col2.metric("Movies", len(df[df['type'] == 'Movie']))
col3.metric("TV Shows", len(df[df['type'] == 'TV Show']))
col4.metric("Avg Strat Score", f"{df['content_score'].mean():.2f}")
col5.metric("Data Reliability", f"{data_reliability:.2%}")

st.divider()

# 🧠 Strategic Insight Engine Panel
st.subheader("💡 Strategic Insights (Prescriptive)")
insights = generate_insights(df, df_raw)
cols = st.columns(len(insights) if insights else 1)
for i, (text, conf) in enumerate(insights):
    conf_class = "confidence-high" if conf == "High" else "confidence-med" if conf == "Medium" else "confidence-low"
    cols[i % len(cols)].info(f"**{text}**\n\nConfidence: <span class='{conf_class}'>{conf}</span>", icon="ℹ️")

st.divider()

# Main UI Tabs
tab_core, tab_advanced, tab_prescriptive = st.tabs(["📊 Core Intelligence", "🧠 Behavioral Analysis", "🎯 Strategic Opportunities"])

with tab_core:
    # 🚀 Growth Intelligence Engine
    c1, c2 = st.columns(2)
    with c1:
        added_trend = df['year_added'].value_counts().sort_index().reset_index()
        added_trend.columns = ['Year', 'Count']
        fig_growth = px.line(added_trend, x='Year', y='Count', title="Content Added Over Time",
                             line_shape='spline', color_discrete_sequence=['#E50914'])
        fig_growth.update_layout(template="plotly_dark")
        st.plotly_chart(fig_growth, use_container_width=True)
    with c2:
        top_countries = df['country'].value_counts().head(10).reset_index()
        top_countries.columns = ['Country', 'Count']
        fig_geo = px.bar(top_countries, x='Count', y='Country', orientation='h', 
                         title="Top Production Hubs", color='Count', color_continuous_scale='Reds')
        fig_geo.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_geo, use_container_width=True)

    st.divider()
    
    # Genre Distribution
    c3, c4 = st.columns(2)
    with c3:
        genres = df['listed_in'].str.split(', ', expand=True).stack().value_counts().head(10).reset_index()
        genres.columns = ['Genre', 'Count']
        fig_genre = px.pie(genres, values='Count', names='Genre', title="Top Genre distribution",
                           hole=0.4, color_discrete_sequence=px.colors.sequential.Reds_r)
        fig_genre.update_layout(template="plotly_dark")
        st.plotly_chart(fig_genre, use_container_width=True)
    with c4:
        st.markdown("### Audience Distribution (Rating)")
        ratings = df['rating'].value_counts().head(10).reset_index()
        ratings.columns = ['Rating', 'Count']
        fig_rate = px.bar(ratings, x='Rating', y='Count', color='Count', color_continuous_scale='Reds')
        fig_rate.update_layout(template="plotly_dark")
        st.plotly_chart(fig_rate, use_container_width=True)

with tab_advanced:
    st.subheader("🧠 Behavioral & Structural Intelligence")
    ac1, ac2 = st.columns(2)
    with ac1:
        # Geographical Expansion
        expansion = df.groupby('year_added')['country'].nunique().reset_index()
        expansion.columns = ['Year', 'Unique Countries']
        fig_exp = px.area(expansion, x='Year', y='Unique Countries', title="Geographical Market Expansion",
                          color_discrete_sequence=['#E50914'])
        fig_exp.update_layout(template="plotly_dark")
        st.plotly_chart(fig_exp, use_container_width=True)

    with ac2:
        # Content Mix Evolution
        mix = df.groupby(['year_added', 'type']).size().unstack().fillna(0).reset_index()
        fig_mix = px.bar(mix, x='year_added', y=['Movie', 'TV Show'], title="Content Mix Evolution",
                         color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'})
        fig_mix.update_layout(template="plotly_dark", barmode='stack')
        st.plotly_chart(fig_mix, use_container_width=True)

    st.divider()

    ac3, ac4 = st.columns(2)
    with ac3:
        # Genre Diversity Index
        genre_exp = df.copy()
        genre_exp['genre_list'] = genre_exp['listed_in'].str.split(', ')
        diversity = genre_exp.explode('genre_list').groupby('year_added')['genre_list'].nunique().reset_index()
        fig_div = px.line(diversity, x='year_added', y='genre_list', title="Genre Diversity Index",
                          line_shape='hv', color_discrete_sequence=['#E50914'])
        fig_div.update_layout(template="plotly_dark")
        st.plotly_chart(fig_div, use_container_width=True)

    with ac4:
        # Content Lifecycle analysis
        fig_life = px.histogram(df, x='content_age', nbins=20, title="Content Lifecycle (Age when Added)",
                                color_discrete_sequence=['#E50914'])
        fig_life.update_layout(template="plotly_dark")
        st.plotly_chart(fig_life, use_container_width=True)

with tab_prescriptive:
    st.subheader("🎯 Strategic Opportunities & Decision Support")
    
    # 🌍 3. Market Opportunity Detection
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("### Emerging Market Opportunities")
        country_growth = df.groupby('country')['year_added'].mean().sort_values(ascending=False).head(10).reset_index()
        country_growth.columns = ['Country', 'Avg Year Added']
        fig_emerge = px.bar(country_growth, x='Avg Year Added', y='Country', orientation='h', 
                            title="Fastest Growing Sourcing Countries (Late-Entry Growth)",
                            color='Avg Year Added', color_continuous_scale='Reds', range_x=[2015, 2024])
    st.plotly_chart(fig_emerge, use_container_width=True)
    st.info("💡 Identifies regions where Netflix has recently accelerated content acquisition.")

    with pc2:
        # 🎯 2. Top Performing Segments
        st.markdown("### High-Value Strategic Segments")
        segment = df.groupby(['type', 'rating'])['content_score'].mean().sort_values(ascending=False).head(8).reset_index()
        fig_seg = px.bar(segment, x='content_score', y='rating', color='type', 
                         title="Avg Strategy Score by Segment (Type + Rating)", barmode='group',
                         color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'})
        fig_seg.update_layout(template="plotly_dark")
        st.plotly_chart(fig_seg, use_container_width=True)
        st.info("💡 Ranks content categories by strategic importance (Rating + Recency + Format).")

    st.divider()

    # 🕵️ 5. Outlier Intelligence
    pc3, pc4 = st.columns(2)
    with pc3:
        st.markdown("### Outlier Intelligence: Extreme Durations")
        outliers = df[df['duration_num'] > 200].sort_values(by='duration_num', ascending=False).head(10)
        st.dataframe(outliers[['title', 'type', 'duration_num', 'country']], use_container_width=True)
        st.warning(f"Detected {len(df[df['duration_num'] > 200])} titles with extreme durations (>200 units).")

    with pc4:
        st.markdown("### Strategic Content Catalog (Top Rated)")
        st.dataframe(df.sort_values(by='content_score', ascending=False)[['title', 'type', 'release_year', 'content_score']].head(10), use_container_width=True)

# -------------------------------
# 🏁 Footer
# -------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; color: gray;">
        NCIP: Netflix Content Intelligence Platform | Prescriptive Decision Edition | Built for Strategic Data Excellence
    </div>
    """, unsafe_allow_html=True)
