import sys
sys.path.insert(0, '/Users/sudhinkarki/Desktop/Hackathon2/klqe')

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# --- Set Page Layout ---
st.set_page_config(page_title="KL Curation Dashboard", layout="wide")
st.title("🧠 Knowledge Library Curation Dashboard")

# --- Load Data from CSV ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/suggestions.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["deleted_at"], errors="coerce")  # assuming deleted_at is last update or deprecate date

    # Ensure needed columns exist
    if 'product_name' not in df.columns:
        df['product_name'] = "Unknown"
    if 'category' not in df.columns:
        df['category'] = "Uncategorized"

    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Overview", "🔍 Clusters", "🆕 Outdated", "🔁 Compare"])

# --- Mock Cluster IDs (for demo purposes) ---
np.random.seed(42)
df['cluster_id'] = pd.qcut(df.groupby('product_id').cumcount(), q=10, duplicates='drop').cat.codes
df['cluster_id'] = df['cluster_id'].astype(int)

# --- Generate Mock Outdated Flag ---
six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
df['is_outdated'] = df['created_at'] < six_months_ago

# --- Fake Suggestions for Merging ---
merge_suggestions = {
    1: {"primary": df.iloc[0]['cqid'], "merge_with": [df.iloc[1]['cqid'], df.iloc[2]['cqid']]},
    2: {"primary": df.iloc[3]['cqid'], "merge_with": [df.iloc[4]['cqid']]},
}

# --- Main Pages Logic ---

if page == "🏠 Home":
    st.subheader("🏡 Welcome to KL Curation Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Products", df['product_id'].nunique())
        st.metric("Total Categories", df['category'].nunique())

        prod_counts = df['product_name'].value_counts().reset_index()
        prod_counts.columns = ['product_name', 'count']
        fig = px.bar(prod_counts, x='product_name', y='count', title="📚 Questions per Product")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Total Q&A Entries", len(df))
        st.metric("Outdated Candidates", df['is_outdated'].sum())

        cat_counts = df['category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'count']
        fig = px.pie(cat_counts, names='category', values='count', title="🏷️ Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📌 Recent Q&A Entries")
    st.dataframe(df[['question', 'product_name', 'category', 'created_at']].sort_values(by='created_at', ascending=False).head(10))

elif page == "📊 Overview":
    st.subheader("📈 Product & Category Analysis")

    selected_product = st.selectbox("Select Product", options=["All"] + list(df['product_name'].unique()))
    filtered_df = df if selected_product == "All" else df[df['product_name'] == selected_product]

    # Product or All?
    if selected_product != "All":
        st.markdown(f"#### Selected Product: {selected_product}")
        col1, col2 = st.columns(2)

        with col1:
            cat_counts = filtered_df['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            fig = px.bar(cat_counts, x='category', y='count', title="Category Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(filtered_df, names='is_outdated', title="Outdated vs Active")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔎 Q&A List")
        st.dataframe(filtered_df[['cqid', 'question', 'answer', 'category', 'created_at']])
    else:
        st.markdown("#### Summary Across All Products")

        cat_counts = df.groupby('category').size().reset_index(name='total')
        fig = px.bar(cat_counts, x='category', y='total', title="Total Q&A by Category")
        st.plotly_chart(fig, use_container_width=True)

        prod_cat_pivot = pd.crosstab(index=df['product_name'], columns=df['category'], margins=True)
        st.markdown("#### Q&A Count by Product and Category")
        st.dataframe(prod_cat_pivot.style.background_gradient(axis=None))

elif page == "🔍 Clusters":
    st.subheader("🔗 Similar Question Groups")

    valid_clusters = df['cluster_id'].unique()
    selected_cluster = st.selectbox("Select Cluster ID", options=valid_clusters)

    cluster_qas = df[df['cluster_id'] == selected_cluster]
    st.write(f"Found {len(cluster_qas)} questions in this cluster:")

    st.dataframe(cluster_qas[['cqid', 'product_name', 'question', 'answer', 'created_at']])

    if selected_cluster in merge_suggestions:
        suggestion = merge_suggestions[selected_cluster]
        st.success("✅ Suggested Merge:")
        st.json(suggestion)
    else:
        st.info("No merge suggestion available for this cluster.")

elif page == "🆕 Outdated":
    st.subheader("🕘 Potentially Outdated Entries")

    outdated_qas = df[df['is_outdated']]
    st.write(f"Found {len(outdated_qas)} potentially outdated Q&A pairs.")

    st.dataframe(outdated_qas[['cqid', 'question', 'product_name', 'category', 'created_at', 'updated_at']])

elif page == "🔁 Compare":
    st.subheader("🔄 Compare Two Q&A Entries")

    ids = df['cqid'].unique().tolist()
    id1 = st.selectbox("Choose First cqid", ids, key="id1")
    id2 = st.selectbox("Choose Second cqid", ids, key="id2")

    row1 = df[df['cqid'] == id1].iloc[0]
    row2 = df[df['cqid'] == id2].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### cqid: {id1}")
        st.markdown("**Product:** " + str(row1['product_name']))
        st.markdown("**Category:** " + str(row1['category']))
        st.markdown("**Question:** " + row1['question'])
        st.markdown("**Answer:** " + row1['answer'])
        st.markdown(f"Created At: {row1['created_at']}")

    with col2:
        st.markdown(f"#### cqid: {id2}")
        st.markdown("**Product:** " + str(row2['product_name']))
        st.markdown("**Category:** " + str(row2['category']))
        st.markdown("**Question:** " + row2['question'])
        st.markdown("**Answer:** " + row2['answer'])
        st.markdown(f"Created At: {row2['created_at']}")

    if st.button("Suggest Merge"):
        st.success("✅ Suggested action: Merge cqid {} into cqid {}".format(min(id1, id2), max(id1, id2)))

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Dashboard built with Streamlit. Ready for AI/NLP integration later.")