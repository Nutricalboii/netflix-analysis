import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    
    # 🎯 Content Strategy Scoring
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
# 🤖 AI Engine: Recommender (Data Science Phase)
# -------------------------------
@st.cache_resource
def get_recommender(data):
    data = data.copy()
    data['metadata_soup'] = data['type'] + " " + data['listed_in'] + " " + data['rating'] + " " + data['description']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['metadata_soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = get_recommender(df_raw)
indices = pd.Series(df_raw.index, index=df_raw['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, df=df_raw):
    if title not in indices: return pd.DataFrame(), []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'type', 'listed_in', 'description']], [i[1] for i in sim_scores]

# -------------------------------
# 🧬 AI Engine: Market Segmentation (Unsupervised Phase)
# -------------------------------
@st.cache_data
def get_market_segments(data):
    # Aggregate by country
    country_stats = data.groupby('country').agg({
        'title': 'count',
        'content_score': 'mean',
        'year_added': 'mean'
    }).reset_index()
    
    # TV Show Ratio
    tv_ratio = data[data['type'] == 'TV Show'].groupby('country').size() / data.groupby('country').size()
    country_stats['tv_ratio'] = country_stats['country'].map(tv_ratio).fillna(0)
    
    country_stats.columns = ['country', 'total_titles', 'avg_score', 'avg_recency', 'tv_ratio']
    
    # Preprocessing for Clustering
    features = ['total_titles', 'avg_score', 'tv_ratio', 'avg_recency']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(country_stats[features])
    
    # KMeans Clustering (k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    country_stats['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Archetype Mapping
    def label_archetype(row):
        if row['total_titles'] > 100: return "Global Powerhouse"
        if row['tv_ratio'] > 0.5: return "Binge-Culture Hub"
        if row['avg_score'] > 3.0: return "Strategic High-Value Hub"
        return "Emerging Specialist"
    
    country_stats['archetype'] = country_stats.apply(label_archetype, axis=1)
    return country_stats

market_df = get_market_segments(df_raw)

# -------------------------------
# 🧠 Strategic Insight Engine (with Confidence)
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
    if counts.get('Movie', 0) > counts.get('TV Show', 0):
        insights.append(("Movies dominate the platform share.", get_conf()))
    else:
        insights.append(("TV Shows lead this segment (Binge-culture focus).", get_conf()))
        
    if not filtered_df['year_added'].empty:
        max_year = filtered_df['year_added'].value_counts().idxmax()
        if max_year > 2015:
            insights.append(("Strategic pivot to 'Originals' detected post-2015.", get_conf()))

    top_rating = filtered_df['rating'].value_counts().idxmax()
    if top_rating in ['TV-MA', 'R']:
        insights.append((f"Adult audience ({top_rating}) is the primary target.", get_conf()))

    avg_score = filtered_df['content_score'].mean()
    if avg_score > 3.0:
        insights.append(("Strategic Content Score is peaking in this segment.", "High"))

    return insights

# -------------------------------
# 🛠️ Sidebar Filters
# -------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
st.sidebar.title("Intelligence Filters")

content_types = st.sidebar.multiselect("Select Content Type", options=df_raw['type'].unique(), default=df_raw['type'].unique())
min_year, max_year = int(df_raw['release_year'].min()), int(df_raw['release_year'].max())
year_range = st.sidebar.slider("Release Year Range", min_value=min_year, max_value=max_year, value=(2010, max_year))
countries = sorted(df_raw['country'].unique())
selected_countries = st.sidebar.multiselect("Select Countries", options=countries, default=["United States", "India", "United Kingdom", "South Korea", "Japan"])

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
        fig_growth = px.line(added_trend, x='Year', y='Count', title="Content Added Over Time", line_shape='spline', color_discrete_sequence=['#E50914'])
        fig_growth.update_layout(template="plotly_dark")
        st.plotly_chart(fig_growth, use_container_width=True)
    with c2:
        top_countries_df = df['country'].value_counts().head(10).reset_index()
        top_countries_df.columns = ['Country', 'Count']
        fig_geo = px.bar(top_countries_df, x='Count', y='Country', orientation='h', title="Top Production Hubs", color='Count', color_continuous_scale='Reds')
        fig_geo.update_layout(template="plotly_dark")
        st.plotly_chart(fig_geo, use_container_width=True)

with tab_advanced:
    st.subheader("🧠 Behavioral & Structural Intelligence")
    ac1, ac2 = st.columns(2)
    with ac1:
        expansion = df.groupby('year_added')['country'].nunique().reset_index()
        expansion.columns = ['Year', 'Unique Countries']
        st.plotly_chart(px.area(expansion, x='Year', y='Unique Countries', title="Geographical expansion", color_discrete_sequence=['#E50914']).update_layout(template="plotly_dark"), use_container_width=True)
    with ac2:
        mix = df.groupby(['year_added', 'type']).size().unstack().fillna(0).reset_index()
        st.plotly_chart(px.bar(mix, x='year_added', y=['Movie', 'TV Show'], title="Content Mix Evolution", color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'}).update_layout(template="plotly_dark"), use_container_width=True)

with tab_prescriptive:
    st.subheader("🎯 Strategic Opportunities & Decision Support")
    pc1, pc2 = st.columns(2)
    with pc1:
        country_growth = df.groupby('country')['year_added'].mean().sort_values(ascending=False).head(10).reset_index()
        country_growth.columns = ['Country', 'Avg Year Added']
        st.plotly_chart(px.bar(country_growth, x='Avg Year Added', y='Country', orientation='h', title="Fastest Growing Sourcing Countries", color='Avg Year Added', color_continuous_scale='Reds', range_x=[min(max_year-5, 2015), max_year]).update_layout(template="plotly_dark"), use_container_width=True)
    with pc2:
        segment = df.groupby(['type', 'rating'])['content_score'].mean().sort_values(ascending=False).head(8).reset_index()
        st.plotly_chart(px.bar(segment, x='content_score', y='rating', color='type', title="Avg Strategy Score by Segment", barmode='group', color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'}).update_layout(template="plotly_dark"), use_container_width=True)

with tab_ai:
    st.subheader("🤖 SmartContent AI Recommender")
    target_title = st.selectbox("Search for a Title to see Recommendations", options=sorted(df_raw['title'].unique()))
    if target_title:
        rec_df, scores = get_recommendations(target_title)
        st.markdown(f"### Top 5 Recommendations for: **{target_title}**")
        cols = st.columns(5)
        for i, (idx, row) in enumerate(rec_df.iterrows()):
            with cols[i]:
                st.markdown(f"**{row['title']}**")
                st.caption(f"Match: {scores[i]:.0%}")
                st.write(row['description'][:100] + "...")

with tab_seg:
    st.subheader("🧬 Market Segmentation (Unsupervised Learning)")
    st.markdown("Countries grouped into strategic clusters based on production volume, strategy scores, and format preferences.")
    
    # 3D Market Cluster Plot
    fig_3d = px.scatter_3d(market_df, x='total_titles', y='tv_ratio', z='avg_score',
                           color='archetype', size='total_titles', hover_name='country',
                           title="Global Market Sourcing Archetypes (3D Cluster Analysis)",
                           labels={'total_titles': 'Volume', 'tv_ratio': 'TV Show Ratio', 'avg_score': 'Strat Score'},
                           color_discrete_sequence=px.colors.qualitative.Reds)
    fig_3d.update_layout(template="plotly_dark", scene=dict(bgcolor='#141414'))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.divider()
    
    # Archetype Breakdown
    st.markdown("### Strategic Archetype Definitions")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.write("**Global Powerhouses**: Volume leaders with balanced format strategy (e.g., USA, India).")
        st.write("**Binge-Culture Hubs**: Countries with a dominant focus on TV Series production.")
    with m_col2:
        st.write("**Strategic High-Value Hubs**: Producers of high-recency, high-rated strategic content.")
        st.write("**Emerging Specialists**: Late-entry markets with developing production depth.")

# -------------------------------
# 🏁 Footer
# -------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; color: gray;">
        NCIP: Netflix Content Intelligence Platform | Unsupervised Learning Edition | Built for Strategic Data Excellence
    </div>
    """, unsafe_allow_html=True)
