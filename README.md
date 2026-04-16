# 🎬 NCIP: Netflix Content Intelligence Platform

The **Netflix Content Intelligence Platform (NCIP)** is an elite data intelligence system that transforms raw streaming data into strategic business decisions. This version transitions the platform into **Data Science** by integrating an AI-powered recommendation engine.

## 🚀 Strategic Vision
NCIP shifts from simple "data plots" to **AI-Integrated Intelligence Systems**. It bridges the gap between raw entertainment data and platform strategy, enabling real-time exploration of global content trends supported by Machine Learning models.

---

## 🤖 Data Science & AI Layer

### 1. SmartContent AI Recommender
A high-performance recommendation engine built using **Natural Language Processing (NLP)**.
- **Technology**: TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization and Cosine Similarity.
- **Analysis**: The system builds a metadata "soup" for every title (Description + Genre + Rating + Format) and calculates historical similarity to suggest content.
- **Interactivity**: Users can search for any title to discover the Top 5 most semantically related content pieces with calculated "Match Confidence %".

---

## 🧠 Intelligence Layers

### 1. Prescriptive Strategy Engine
- **Strategic Content Scoring**: A proprietary algorithm that ranks every title based on format, audience rating, and market recency.
- **Insight Confidence Tagging**: High/Medium/Low confidence markers on every insight, dynamically calculated based on data density.
- **Data Reliability Index**: A top-level KPI quantifying the trust-level of the underlying dataset.

### 2. Behavioral Intelligence
- **Market Expansion Tracking**: Monitoring Netflix's geographical scaling by tracking unique country sourcing over time.
- **Content Lifecycle Analysis**: Analyzing "New Releases" vs "Library Depth".

---

## 🛠️ Technology Stack
- **AI/ML**: [Scikit-learn](https://scikit-learn.org/) (NLP & Similarity Models)
- **UI Layer**: [Streamlit](https://streamlit.io/) (Premium Dark Mode Interface)
- **Viz Engine**: [Plotly](https://plotly.com/python/) & [Seaborn](https://seaborn.pydata.org/)
- **Data Engine**: [Pandas](https://pandas.pydata.org/)

---

## 📁 Project Architecture
```text
netflix-analysis/
├── app.py                # NCIP AI Dashboard (Main UI)
├── netflix.csv           # Global Content Intelligence Source
├── analysis.py           # Legacy Script (Logic Audit)
├── README.md             # Project & AI Strategy Documentation
└── tests/                # Data Integrity & Validation Suite
```

---

## 🚀 Launching the Platform

### 1. Install Dependencies
Ensure you have the full AI intelligence stack:
```bash
pip install streamlit pandas plotly seaborn scikit-learn
```

### 2. Run the Dashboard
Power up the NCIP platform locally:
```bash
streamlit run app.py
```

---
*Built for Strategic Data Excellence. This project represents the transition from Data Intelligence to Data Science.*
