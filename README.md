# 🎬 NCIP: Netflix Content Intelligence Platform

The **Netflix Content Intelligence Platform (NCIP)** is a high-performance data intelligence system that transforms Netflix's global dataset into actionable strategic insights via an interactive visual dashboard.

## 🚀 Vision
NCIP shifts from static analysis to **Intelligence Systems Thinking**. It bridges the gap between raw entertainment data and platform strategy, enabling real-time exploration of global content trends.

---

## 📊 Core Intelligence Modules

### 📈 Platform Growth Intelligence
Tracks Netflix's scaling strategy by analyzing content acquisition trends over time.
- **Content Added**: Platform scaling pace.
- **Release Trend**: Historical catalog depth.

### 🌍 Geographical Intelligence
Identifies lead content-producing nations and tracks the shift toward international production hubs like India and South Korea.

### 🎭 Genre & Audience Intelligence
Explores content diversity and audience targeting strategies through:
- **Genre Distribution**: Mapping content backbone.
- **Rating Heatmaps**: Visualizing audience targeting (e.g., adult-oriented focus).

### 🎯 Duration Intelligence
Structural analysis of standard content lengths to identify production norms.

---

## 🛠️ Technology Stack
- **UI Layer**: [Streamlit](https://streamlit.io/) (Premium Dark Mode Interface)
- **Viz Engine**: [Plotly](https://plotly.com/python/) (Interactive Visuals) & [Seaborn](https://seaborn.pydata.org/)
- **Data Engine**: [Pandas](https://pandas.pydata.org/)
- **Styling**: Netflix-Identity (Red/Black) custom CSS

---

## 📁 Project Architecture
```text
netflix-analysis/
├── app.py                # NCIP Dashboard Engine (Main UI)
├── netflix.csv           # Global Content Dataset
├── analysis.py           # Legacy Advanced Analysis Script
├── README.md             # Product Documentation
└── plots/                # High-res static exports
```

---

## 🚀 Launching the Platform

### 1. Install Dependencies
Ensure you have the intelligence stack installed:
```bash
pip install streamlit pandas plotly seaborn matplotlib
```

### 2. Run the Dashboard
Power up the NCIP platform locally:
```bash
streamlit run app.py
```

---
*Built for Strategic Data Excellence.*
