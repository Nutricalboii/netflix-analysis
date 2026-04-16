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
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 📂 Data Intelligence Engine
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('netflix.csv')
    # Preprocessing
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df = df.dropna(subset=['type', 'country', 'release_year', 'rating'])
    return df

df_raw = load_data()

# -------------------------------
# 🛠️ Sidebar Filters
# -------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
st.sidebar.title("Intelligence Filters")

# Filter: Content Type
content_type = st.sidebar.multiselect(
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
    default=["United States", "India", "United Kingdom"]
)

# Apply Filters
df = df_raw[
    (df_raw['type'].isin(content_type)) &
    (df_raw['release_year'].between(year_range[0], year_range[1])) &
    (df_raw['country'].isin(selected_countries))
]

# -------------------------------
# 📊 Dashboard Header
# -------------------------------
st.title("🎬 Netflix Content Intelligence Platform (NCIP)")
st.markdown("Transforming raw streaming data into actionable strategic insights.")

# Key Performance Indicators (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", len(df))
col2.metric("Movies", len(df[df['type'] == 'Movie']))
col3.metric("TV Shows", len(df[df['type'] == 'TV Show']))
col4.metric("Avg Duration (Min)", f"{df[df['type'] == 'Movie']['duration_num'].mean():.1f}" if not df[df['type'] == 'Movie'].empty else "N/A")

st.divider()

# -------------------------------
# 🚀 Growth Intelligence Engine
# -------------------------------
st.subheader("📈 Platform Growth Intelligence")
tab1, tab2 = st.tabs(["Content Added (Platform Growth)", "Release Trend (Production Growth)"])

with tab1:
    added_trend = df['year_added'].value_counts().sort_index().reset_index()
    added_trend.columns = ['Year', 'Count']
    fig_growth = px.line(added_trend, x='Year', y='Count', title="Titles Added to Netflix Over Time",
                         line_shape='spline', color_discrete_sequence=['#E50914'])
    fig_growth.update_layout(template="plotly_dark")
    st.plotly_chart(fig_growth, use_container_width=True)
    st.info("💡 **Insight:** This reflects Netflix's platform scaling and acquisition strategy.")

with tab2:
    release_trend = df['release_year'].value_counts().sort_index().reset_index()
    release_trend.columns = ['Year', 'Count']
    fig_release = px.area(release_trend, x='Year', y='Count', title="Content Production History",
                          color_discrete_sequence=['#E50914'])
    fig_release.update_layout(template="plotly_dark")
    st.plotly_chart(fig_release, use_container_width=True)
    st.info("💡 **Insight:** Tracks industry-wide production spikes and historical catalog depth.")

# -------------------------------
# 🌍 Geographical & Genre Intelligence
# -------------------------------
st.divider()
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🌍 Geographical Intelligence")
    top_countries = df['country'].value_counts().head(10).reset_index()
    top_countries.columns = ['Country', 'Count']
    fig_geo = px.bar(top_countries, x='Count', y='Country', orientation='h', 
                     title="Top Content Producing Hubs", color='Count', color_continuous_scale='Reds')
    fig_geo.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_geo, use_container_width=True)

with right_col:
    st.subheader("🎭 Genre Intelligence")
    genres = df['listed_in'].str.split(', ', expand=True).stack().value_counts().head(10).reset_index()
    genres.columns = ['Genre', 'Count']
    fig_genre = px.pie(genres, values='Count', names='Genre', title="Top 10 Genre Distribution",
                       hole=0.4, color_discrete_sequence=px.colors.sequential.Reds_r)
    fig_genre.update_layout(template="plotly_dark")
    st.plotly_chart(fig_genre, use_container_width=True)

# -------------------------------
# 🧠 Audience Targeting Analysis
# -------------------------------
st.divider()
st.subheader("🎯 Audience Targeting & Duration Analysis")
l_col, r_col = st.columns(2)

with l_col:
    st.markdown("### Content Type vs Rating Heatmap")
    pivot = df.pivot_table(index='type', columns='rating', aggfunc='size', fill_value=0)
    top_ratings = df['rating'].value_counts().head(8).index
    pivot_filtered = pivot[top_ratings]
    
    fig_hm, ax_hm = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot_filtered, annot=True, fmt='d', cmap='Reds', ax=ax_hm)
    ax_hm.set_title("Targeting Strategy: Type vs Rating")
    fig_hm.patch.set_facecolor('#141414')
    ax_hm.set_facecolor('#141414')
    st.pyplot(fig_hm)
    st.info("💡 **Insight:** TV-MA dominance suggests a strong focus on adult audiences.")

with r_col:
    st.markdown("### Movie Duration Distribution")
    movie_durations = df[df['type'] == 'Movie']['duration_num'].dropna()
    fig_dur = px.histogram(movie_durations, nbins=30, title="Distribution of Movie Runtimes",
                           color_discrete_sequence=['#E50914'], labels={'value': 'Duration (Min)'})
    fig_dur.update_layout(template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_dur, use_container_width=True)
    st.info("💡 **Insight:** Most movies fall within the standard 90-120 minute range.")

# -------------------------------
# 🏁 Footer
# -------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; color: gray;">
        NCIP: Netflix Content Intelligence Platform | Built for Strategic Data Excellence
    </div>
    """, unsafe_allow_html=True)
