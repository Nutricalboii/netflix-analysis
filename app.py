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
    df['content_age'] = df['year_added'] - df['release_year']
    df = df.dropna(subset=['type', 'country', 'release_year', 'rating'])
    return df

df_raw = load_data()

# -------------------------------
# 🧠 Strategic Insight Engine
# -------------------------------
def generate_insights(filtered_df):
    insights = []
    
    if filtered_df.empty:
        return ["No data available for the selected filters."]
    
    # Type Dominance
    counts = filtered_df['type'].value_counts()
    if counts.get('Movie', 0) > counts.get('TV Show', 0):
        insights.append("Movies currently dominate the platform strategy in this segment.")
    else:
        insights.append("TV Shows have a stronger presence here, highlighting a 'binge-culture' focus.")
        
    # Growth Spikes
    if not filtered_df['year_added'].empty:
        max_year = filtered_df['year_added'].value_counts().idxmax()
        if max_year > 2015:
            insights.append(f"Most content in this view was added after 2015, marking a post-Originals scaling era.")

    # Audience Focus
    top_rating = filtered_df['rating'].value_counts().idxmax()
    if top_rating in ['TV-MA', 'R']:
        insights.append(f"Audience focus is heavily oriented toward mature viewers ({top_rating}).")

    # Geographical Dominance
    top_country = filtered_df['country'].value_counts().idxmax()
    insights.append(f"{top_country} is the primary production hub for this selection.")

    # Diversity Index
    genre_counts = filtered_df['listed_in'].str.split(', ').explode()
    diversity_score = genre_counts.nunique()
    if diversity_score > 15:
        insights.append(f"High Genre Diversity Index ({diversity_score} tags detected), suggesting a varied content strategy.")

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
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", len(df))
col2.metric("Movies", len(df[df['type'] == 'Movie']))
col3.metric("TV Shows", len(df[df['type'] == 'TV Show']))
col4.metric("Unique Countries", df['country'].nunique())

st.divider()

# 🧠 Strategic Insight Engine Panel
st.subheader("💡 Strategic Insights")
insights = generate_insights(df)
cols = st.columns(len(insights) if insights else 1)
for i, insight in enumerate(insights):
    cols[i % len(cols)].info(f"**{insight}**")

st.divider()

# Main UI Tabs
tab_core, tab_advanced = st.tabs(["📊 Core Intelligence", "🧠 Advanced Behavioral Analytics"])

with tab_core:
    # 🚀 Growth Intelligence Engine
    st.subheader("📈 Platform Growth Intelligence")
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
    
    # Genre & Audience
    c3, c4 = st.columns(2)
    with c3:
        genres = df['listed_in'].str.split(', ', expand=True).stack().value_counts().head(10).reset_index()
        genres.columns = ['Genre', 'Count']
        fig_genre = px.pie(genres, values='Count', names='Genre', title="Top Genre Distribution",
                           hole=0.4, color_discrete_sequence=px.colors.sequential.Reds_r)
        fig_genre.update_layout(template="plotly_dark")
        st.plotly_chart(fig_genre, use_container_width=True)
    with c4:
        st.markdown("### Type vs Rating Heatmap")
        pivot = df.pivot_table(index='type', columns='rating', aggfunc='size', fill_value=0)
        top_ratings = df['rating'].value_counts().head(8).index
        pivot_filtered = pivot[top_ratings] if not pivot.empty else pd.DataFrame()
        
        if not pivot_filtered.empty:
            fig_hm, ax_hm = plt.subplots(figsize=(10, 4))
            sns.heatmap(pivot_filtered, annot=True, fmt='d', cmap='Reds', ax=ax_hm)
            fig_hm.patch.set_facecolor('#141414')
            ax_hm.set_facecolor('#141414')
            ax_hm.set_title("Targeting Strategy", color='white')
            st.pyplot(fig_hm)
        else:
            st.warning("Insufficient data for heatmap.")

with tab_advanced:
    st.subheader("🧠 Behavioral & Structural Intelligence")
    
    # 1. Market Expansion & Content Mix
    ac1, ac2 = st.columns(2)
    with ac1:
        # Geographical Expansion
        expansion = df.groupby('year_added')['country'].nunique().reset_index()
        expansion.columns = ['Year', 'Unique Countries']
        fig_exp = px.area(expansion, x='Year', y='Unique Countries', title="Geographical Market Expansion",
                          color_discrete_sequence=['#E50914'])
        fig_exp.update_layout(template="plotly_dark")
        st.plotly_chart(fig_exp, use_container_width=True)
        st.info("💡 Tracks how many unique nations Netflix sources from per year.")

    with ac2:
        # Content Mix Evolution
        mix = df.groupby(['year_added', 'type']).size().unstack().fillna(0).reset_index()
        fig_mix = px.bar(mix, x='year_added', y=['Movie', 'TV Show'], title="Content Mix Evolution",
                         color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'})
        fig_mix.update_layout(template="plotly_dark", barmode='stack')
        st.plotly_chart(fig_mix, use_container_width=True)
        st.info("💡 Visualizes the shift between film and binge-ready TV shows.")

    st.divider()

    # 2. Genre Diversity & Lifecycle
    ac3, ac4 = st.columns(2)
    with ac3:
        # Genre Diversity Index
        genre_exp = df.copy()
        genre_exp['genre_list'] = genre_exp['listed_in'].str.split(', ')
        diversity = genre_exp.explode('genre_list').groupby('year_added')['genre_list'].nunique().reset_index()
        fig_div = px.line(diversity, x='year_added', y='genre_list', title="Genre Diversity Index",
                          line_shape='hv', color_discrete_sequence=['#E50914'])
        fig_div.update_layout(template="plotly_dark", yaxis_title="Unique Genres")
        st.plotly_chart(fig_div, use_container_width=True)
        st.info("💡 Measures the 'variety' breadth of the catalog over time.")

    with ac4:
        # Content Lifecycle analysis
        fig_life = px.histogram(df, x='content_age', nbins=20, title="Content Lifecycle (Age when Added)",
                                color_discrete_sequence=['#E50914'], labels={'content_age': 'Years since Release'})
        fig_life.update_layout(template="plotly_dark")
        st.plotly_chart(fig_life, use_container_width=True)
        st.info("💡 Analyzes the ratio of 'New Releases' vs 'Library Depth' (Licensed content).")

# -------------------------------
# 🏁 Footer
# -------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; color: gray;">
        NCIP: Netflix Content Intelligence Platform | Absolute Ceiling Edition | Built for Strategic Data Excellence
    </div>
    """, unsafe_allow_html=True)
