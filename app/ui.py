import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Make sure pipeline artifacts are on PYTHONPATH if needed
sys.path.insert(0, 'klqe')

# --- Page Setup ---
st.set_page_config(page_title="KL Curation Dashboard", layout="wide")
st.title("🧠 Knowledge Library Curation Dashboard")

# --- Load Master Library ---
@st.cache_data
def load_library():
    path = "data/answer_library_with_clusters.csv"
    if not os.path.exists(path):
        st.error(f"Missing clustered library: {path}")
        st.stop()
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["deleted_at"] = pd.to_datetime(df["deleted_at"], errors="coerce")
    # Rename cluster column if needed
    if "cluster" in df.columns and "cluster_id" not in df.columns:
        df.rename(columns={"cluster": "cluster_id"}, inplace=True)
    # Ensure flags
    df["is_outdated"] = df.apply(lambda r: pd.notna(r["deleted_at"]), axis=1)
    return df

# --- Load Suggestions ---
@st.cache_data
def load_suggestions():
    path = "data/final_suggestions.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["action", "cqid", "target_cqid", "cluster"])
    sug = pd.read_csv(path)
    # Normalize column names
    if "cluster" in sug.columns and "cluster_id" not in sug.columns:
        sug.rename(columns={"cluster": "cluster_id"}, inplace=True)
    return sug

# --- Log Actions ---
def log_action(action_type, cqid, target_cqid=None):
    log_path = "data/action_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    action_entry = {
        "timestamp": timestamp,
        "action": action_type,
        "cqid": cqid,
        "target_cqid": target_cqid,
    }
    # Create a DataFrame for the log
    log_df = pd.DataFrame([action_entry])

    # Append to existing log file or create a new one
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, mode='w', header=True, index=False)

# --- Load Logs ---
def load_action_logs():
    log_path = "data/action_log.csv"
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
    else:
        logs = pd.DataFrame(columns=["timestamp", "action", "cqid", "target_cqid"])
    return logs

df = load_library()
suggestion_df = load_suggestions()
print(df.columns)

if "logged_actions" not in st.session_state:
    st.session_state["logged_actions"] = set()

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Overview", "🔍 Clusters", "🆕 Outdated", "💡 Suggestions", "🔁 Compare", "📜 Action Logs"])

# --- Home Page ---
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

# --- Overview ---
elif page == "📊 Overview":
    st.subheader("📈 Product & Category Analysis")
    selected_product = st.selectbox("Select Product", options=["All"] + sorted(df['product_name'].unique()))
    filtered_df = df if selected_product == "All" else df[df['product_name'] == selected_product]
    if selected_product != "All":
        st.markdown(f"#### Selected Product: {selected_product}")
        c1, c2 = st.columns(2)
        with c1:
            cat_counts = filtered_df['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            fig = px.bar(cat_counts, x='category', y='count', title="Category Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(filtered_df, names='is_outdated', title="Outdated vs Active")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🔎 Q&A List")
        st.dataframe(filtered_df[['cqid', 'question', 'answer', 'category', 'created_at']])
    else:
        cat_counts = df.groupby('category').size().reset_index(name='total')
        fig = px.bar(cat_counts, x='category', y='total', title="Total Q&A by Category")
        st.plotly_chart(fig, use_container_width=True)
        pivot = pd.crosstab(index=df['product_name'], columns=df['category'], margins=True)
        st.markdown("#### Q&A Count by Product and Category")
        st.dataframe(pivot.style.background_gradient(axis=None))

# --- Clusters ---
elif page == "🔍 Clusters":
    st.subheader("🔗 Similar Question Groups")
    cluster_ids = sorted(df['cluster_id'].unique())
    selected_cluster = st.selectbox("Select Cluster ID", options=cluster_ids)
    cluster_df = df[df['cluster_id'] == selected_cluster]
    st.write(f"Found {len(cluster_df)} questions in this cluster:")
    st.dataframe(cluster_df[['cqid', 'question', 'answer', 'product_name', 'category', 'created_at']])

# --- Outdated ---
elif page == "🆕 Outdated":
    st.subheader("🕘 Potentially Outdated Entries")
    outdated_df = df[df['is_outdated']]
    st.write(f"Found {len(outdated_df)} potentially outdated Q&A pairs.")
    st.dataframe(outdated_df[['cqid', 'question', 'product_name', 'category', 'created_at', 'deleted_at']])



# --- Suggestions ---
elif page == "💡 Suggestions":
    st.subheader("🤖 Curator Suggestions")

    if suggestion_df.empty:
        st.warning("No suggestions found.")
    else:
        actions = suggestion_df['action'].unique().tolist()
        selected_action = st.selectbox("Filter by Action", ["All"] + actions)

        filtered = suggestion_df if selected_action == "All" else suggestion_df[suggestion_df['action'] == selected_action]

        cluster_ids = sorted(filtered['cluster_id'].dropna().unique())
        selected_cluster = st.selectbox("Filter by Cluster ID", ["All"] + list(map(str, cluster_ids)), key="sug_cluster")

        if selected_cluster != "All":
            filtered = filtered[filtered['cluster_id'].astype(str) == selected_cluster]

        st.write(f"{len(filtered)} suggestions found.")

        for i, row in filtered.iterrows():
            cqid = row['cqid']
            action = row['action']
            target = row.get('target_cqid', None)
            cluster = row.get('cluster_id', 'N/A')
            key_suffix = f"{cqid}_{action}_{i}"

            # Prevent double logging
            if key_suffix in st.session_state["logged_actions"]:
                continue

            qna_row = df[df['cqid'] == cqid]
            if qna_row.empty:
                continue
            qna_row = qna_row.squeeze()

            with st.expander(f"📌 {action.upper()} — CQID: {cqid} | Cluster: {cluster}"):
                st.markdown(f"**Product:** {qna_row.get('product_name', 'N/A')}")
                st.markdown(f"**Category:** {qna_row.get('category', 'N/A')}")
                st.markdown(f"**Question:** {qna_row.get('question', '')}")
                st.markdown(f"**Details:** {qna_row.get('details', '')}")
                st.markdown(f"**Answer:** {qna_row.get('answer', '')}")

                if action in ["merge", "review"] and pd.notna(target):
                    st.markdown("---")
                    st.markdown(f"**Target → CQID: {target}**")
                    target_row = df[df['cqid'] == target]
                    if not target_row.empty:
                        target_row = target_row.squeeze()
                        st.markdown(f"**Target Product:** {target_row.get('product_name', 'N/A')}")
                        st.markdown(f"**Target Category:** {target_row.get('category', 'N/A')}")
                        st.markdown(f"**Target Question:** {target_row.get('question', '')}")
                        st.markdown(f"**Target Details:** {target_row.get('details', '')}")
                        st.markdown(f"**Target Answer:** {target_row.get('answer', '')}")
                    else:
                        st.warning("Target CQID not found in dataset.")

                if action == "archive":
                    st.markdown("---")
                    st.markdown(f"**Created At:** {qna_row.get('created_at', '')}")
                    st.markdown(f"**Deleted At:** {qna_row.get('deleted_at', 'N/A')}")

                col1, col2 = st.columns(2)
                accept = col1.button("✅ Accept", key=f"accept_{key_suffix}")
                reject = col2.button("❌ Reject", key=f"reject_{key_suffix}")

                if accept or reject:
                    log_action(action, cqid, target)
                    st.session_state["logged_actions"].add(key_suffix)
                    st.success(f"{'Accepted' if accept else 'Rejected'} action for CQID {cqid}")

# --- Compare ---
elif page == "🔁 Compare":
    st.subheader("🔄 Compare Two Q&A Entries")
    ids = df['cqid'].tolist()
    id1 = st.selectbox("First CQID", ids, key="cmp1")
    id2 = st.selectbox("Second CQID", ids, key="cmp2")
    r1, r2 = df[df['cqid'] == id1].iloc[0], df[df['cqid'] == id2].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### CQID: {id1}")
        st.markdown(f"**Product:** {r1['product_name']}")
        st.markdown(f"**Category:** {r1['category']}")
        st.markdown(f"**Question:** {r1['question']}")
        st.markdown(f"**Answer:** {r1['answer']}")
        st.markdown(f"Created At: {r1['created_at']}")
    with c2:
        st.markdown(f"#### CQID: {id2}")
        st.markdown(f"**Product:** {r2['product_name']}")
        st.markdown(f"**Category:** {r2['category']}")
        st.markdown(f"**Question:** {r2['question']}")
        st.markdown(f"**Answer:** {r2['answer']}")
        st.markdown(f"Created At: {r2['created_at']}")


# --- Action Logs ---
elif page == "📜 Action Logs":
    st.subheader("📋 Action Log History")

    # 1) Load raw logs
    log_path = "data/action_log.csv"
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
    else:
        st.warning("No actions logged yet.")
        st.stop()

    # 2) Dedupe raw logs
    logs = logs.drop_duplicates(subset=["timestamp", "action", "cqid", "target_cqid"])
    if logs.empty:
        st.warning("No actions logged after dedupe.")
        st.stop()

    # 3) Enrich with library metadata
    #    (ensure df has unique cqid)
    lib_meta = (
        df.drop_duplicates(subset="cqid")
          .set_index("cqid")
          [["product_name", "category", "created_at", "deleted_at", "cluster_id"]]
    )
    logs = logs.merge(
        lib_meta, 
        left_on="cqid", right_index=True, 
        how="left"
    )

    # 4) Enrich with suggestion context
    sug_meta = (
        suggestion_df.drop_duplicates(subset="cqid")
          .set_index("cqid")
          [["target_cqid", "cluster_id"]]
          .rename(columns={
              "cluster_id": "suggested_cluster",
              "target_cqid": "merge_target_cqid"
          })
    )
    logs = logs.merge(
        sug_meta, 
        left_on="cqid", right_index=True, 
        how="left"
    )

    # 5) Display Summary Charts
    st.markdown(f"**Total actions logged:** {len(logs)}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**By Action Type:**")
        st.bar_chart(logs["action"].value_counts())
    with col2:
        st.markdown("**By Product:**")
        st.bar_chart(logs["product_name"].fillna("Unknown").value_counts())
    with col3:
        st.markdown("**By Category:**")
        st.bar_chart(logs["category"].fillna("Uncategorized").value_counts())

    # 6) Detailed Log Table
    st.markdown("### 🔎 Full Action Log")
    st.dataframe(logs.sort_values(by="timestamp", ascending=False))

    # 7) Export enriched log
    csv = logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Enriched Action Log",
        data=csv,
        file_name="action_log_enriched.csv",
        mime="text/csv"
    )


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Dashboard built with Streamlit. Powered by your pipeline modules.")
