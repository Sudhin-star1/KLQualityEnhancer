
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import math
from datetime import datetime
import numpy as np
import uuid
import pickle
import sys

sys.path.insert(0, 'klqe')

# --- Constants ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PAGE_SIZE = 30

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

def paginate_df(df, key, page_size=10):
    total_pages = (len(df) - 1) // page_size + 1
    page_num = st.number_input(f"Page", min_value=1, max_value=total_pages, step=1, key=key)
    start = (page_num - 1) * page_size
    end = start + page_size
    return df.iloc[start:end]


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


# --- Helper Functions ---
def list_saved_files():
    return [
        f
        for f in os.listdir(UPLOAD_DIR)
        if f.endswith(".csv") and not f.startswith(".")
    ]


def save_metadata(filename, df):
    meta = {
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
    }
    with open(os.path.join(UPLOAD_DIR, f"{filename}_meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def delete_file(filename):
    os.remove(os.path.join(UPLOAD_DIR, filename))
    meta_path = os.path.join(UPLOAD_DIR, f"{filename}_meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)


# pagination utility
def paginate_df(df, key):
    total = len(df)
    max_page = math.ceil(total / PAGE_SIZE)
    if f"{key}_page" not in st.session_state:
        st.session_state[f"{key}_page"] = 0
    page = st.session_state[f"{key}_page"]
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Prev", key=f"{key}_prev") and page > 0:
            st.session_state[f"{key}_page"] = page - 1
    with col_info:
        st.markdown(f"Page {page+1} of {max_page}")
    with col_next:
        if st.button("Next →", key=f"{key}_next") and page < max_page - 1:
            st.session_state[f"{key}_page"] = page + 1
    return df.iloc[start:end]


# sanitize object columns for pyarrow
def sanitize_df(df):
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == "object":
            try:
                df2[col] = pd.to_numeric(df2[col], errors="raise")
            except Exception:
                df2[col] = df2[col].astype(str)
    return df2


# Initialize session state keys
for key in ["df", "current_file", "merging", "merged_entries"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "merged_entries" else []

# --- Page Config ---
st.set_page_config(page_title="KL Curation Dashboard", layout="wide")
st.title("🧠 Knowledge Library Curation Dashboard")


df = load_library()
suggestion_df = load_suggestions()
print(df.columns)

if "logged_actions" not in st.session_state:
    st.session_state["logged_actions"] = set()

# --- File Upload & Selection ---
if st.session_state.df is None:
    st.header("Upload or Load CSV Data")
    st.subheader("📤 Upload New File")
    uploaded = st.file_uploader("Choose a CSV", type=["csv"], key="uploader")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df["created_at"] = pd.to_datetime(
                df.get("created_at", None), errors="coerce"
            )
            df.to_csv(os.path.join(UPLOAD_DIR, uploaded.name), index=False)
            save_metadata(uploaded.name, df)
            st.success(f"✅ '{uploaded.name}' uploaded successfully.")
            st.session_state.df = df
            st.session_state.current_file = uploaded.name
            st.rerun()
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")
    st.markdown("---")
    st.subheader("📂 Recent Files")
    files = list_saved_files()
    if files:
        for f in files:
            with st.container():
                c1, c2, c3 = st.columns([6, 1, 1], gap="small")
                c1.markdown(f"**{f}**")
                with c2:
                    if st.button("Load", key=f + "_load"):
                        try:
                            df = pd.read_csv(os.path.join(UPLOAD_DIR, f))
                            df["created_at"] = pd.to_datetime(
                                df.get("created_at", None), errors="coerce"
                            )
                            st.session_state.df = df
                            st.session_state.current_file = f
                            st.rerun()
                        except Exception as e:
                            st.error(f"⚠️ Could not load '{f}': {e}")
                with c3:
                    if st.button("Delete", key=f + "_del"):
                        delete_file(f)
                        st.success(f"🗑️ '{f}' deleted.")
                        st.rerun()
    else:
        st.info("No recent files found.")

else:
    # --- Data Preparation ---
    df = st.session_state.df.copy()
    for col in ["product_name", "category", "cqid", "duplicate_count"]:
        if col not in df.columns:
            df[col] = "Unknown" if col != "duplicate_count" else 0
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["cluster_id"] = pd.qcut(
        df.groupby("product_name").cumcount(), q=10, duplicates="drop"
    ).cat.codes
    # df["is_outdated"] = df["created_at"] < (
    #     pd.Timestamp.now() - pd.DateOffset(months=6)
    # )

    # Sidebar Filters
    st.sidebar.header("Filters")
    keywords = st.sidebar.text_input("Search Question")
    cat = st.sidebar.selectbox("Category", ["All"] + list(df["category"].unique()))
    prod = st.sidebar.selectbox(
        "Product Name", ["All"] + list(df["product_name"].unique())
    )
    date_min = st.sidebar.date_input("Start Date", df["created_at"].min().date())
    date_max = st.sidebar.date_input("End Date", df["created_at"].max().date())
    dup_min, dup_max = st.sidebar.slider(
        "Duplicates",
        0,
        int(df["duplicate_count"].max()),
        (0, int(df["duplicate_count"].max())),
    )
    filt = df.copy()
    if keywords:
        filt = filt[filt["question"].str.contains(keywords, case=False, na=False)]
    if cat != "All":
        filt = filt[filt["category"] == cat]
    if prod != "All":
        filt = filt[filt["product_name"] == prod]
    filt = filt[
        (filt["created_at"].dt.date >= date_min)
        & (filt["created_at"].dt.date <= date_max)
    ]
    filt = filt[
        (filt["duplicate_count"] >= dup_min) & (filt["duplicate_count"] <= dup_max)
    ]


    ## Hello
    df1 = pd.read_csv('data/cleaned_duplicates_product_question_details.csv')
    df1["created_at"] = pd.to_datetime(df1["created_at"],format="mixed", errors='coerce')
    df1["date_only"] = df1["created_at"].dt.date
    time_series = df1.groupby(["date_only", "is_outdated"]).size().unstack(fill_value=0)
    time_series["Cumulative_Outdated"] = time_series[True].cumsum()
    time_series["Cumulative_Current"] = time_series[False].cumsum()

    # Tabs
    tabs = st.tabs(
        [
            "Overview",
            "Clusters",
            "Outdated",
            "Compare",
            "Product Analysis",
            "Category Matrix",
            "Time Trends",
            "Merged Entries",
            "All Entries"
        ]
    )

    with tabs[0]:  # Overview
        st.subheader("🚀 Overview")
        with st.expander("📈 Health Metrics Summary", expanded=True):
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Total Entries", len(df))
            mcol2.metric("Outdated Entries", int(df["is_outdated"].sum()))
            mcol3.metric("Unique Products", df["product_name"].nunique())

            mcol4, mcol5, mcol6 = st.columns(3)
            mcol4.metric("Unique Categories", df["category"].nunique())
            mcol5.metric("Avg Duplicate Count", round(df["duplicate_count"].mean(), 2))
            mcol6.metric(
                "Most Recent Entry", df["created_at"].max().strftime("%Y-%m-%d")
            )

        col1, col2 = st.columns(2)
        with col1:
            cat_df = df["category"].value_counts().reset_index(name="count")
            cat_df.columns = ["category", "count"]

            # Use a vibrant and colorful bar chart
            fig = px.bar(
                cat_df,
                x="category",
                y="count",
                title="Questions per Category",
                height=500,
                color="category",  # Assign different colors based on category
                color_discrete_sequence=px.colors.qualitative.Prism,  # <-- Even more vibrant!
            )

            # Improve layout aesthetics
            fig.update_layout(
                showlegend=False,
                xaxis_title="Category",
                yaxis_title="Number of Questions",
                title_font_size=18,
                font=dict(family="Arial", size=12),
                hovermode="x unified",
                margin=dict(l=20, r=40, t=60, b=60),
            )

            # Optional: Add data labels on top of bars for clarity
            fig.update_traces(texttemplate="%{y}", textposition="outside")

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            counts = df["product_name"].value_counts().reset_index(name="count")
            counts.columns = ["product_name", "count"]

            # Use a vibrant built-in color palette from Plotly
            fig2 = px.pie(
                counts,
                names="product_name",
                values="count",
                title="Product Distribution",
                height=500,
                color_discrete_sequence=px.colors.qualitative.Bold,  # <-- 🔥 Colorful palette
            )
            fig2.update_traces(textinfo="percent+label")  # Show percentage and label
            fig2.update_layout(
                showlegend=False
            )  # Optional: Hide legend if layout is clean enough

            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Entry Creation Timeline")
        fig_time = px.histogram(df, x="created_at", nbins=30, title="Entries Over Time")
        st.plotly_chart(fig_time, use_container_width=True)


# HIDE
    # with tabs[1]:  # Clusters with pagination, merge, flag, archive
    #     st.subheader("🔍 Clusters")
    #     sel_cluster = st.selectbox(
    #         "Select Cluster ID", sorted(df["cluster_id"].unique())
    #     )
    #     cluster_df = filt[filt["cluster_id"] == sel_cluster].reset_index(drop=True)

    #     # Paginate cluster entries
    #     paged = paginate_df(cluster_df, "cluster")

    #     # Truncation helper
    #     def truncate(text, max_len=80):
    #         return text if len(text) <= max_len else text[:max_len] + "..."

    #     # Build question map for multiselect
    #     qmap = {row["cqid"]: truncate(row["question"]) for _, row in paged.iterrows()}
    #     selected = st.multiselect(
    #         "Select entries to Merge",
    #         options=list(qmap.keys()),
    #         format_func=lambda x: f"{x}: {qmap[x]}",
    #         key="sel_multi",
    #     )

    #     # Store selected IDs in session state to persist across pages
    #     if selected:
    #         st.session_state.selected_ids = selected

    #     # Show details for selected entries via expander
    #     if selected:
    #         st.markdown("**Selected Entries Details:**")
    #         for cqid in selected:
    #             row = cluster_df[cluster_df["cqid"] == cqid].iloc[0]
    #             with st.expander(f"CQID {cqid} Details", expanded=False):
    #                 st.write(
    #                     {
    #                         "Product": row["product_name"],
    #                         "Category": row["category"],
    #                         "Question": row["question"],
    #                         "Answer": row.get("answer", ""),
    #                         "Duplicate Count": row["duplicate_count"],
    #                         "Created At": row["created_at"],
    #                     }
    #                 )

    #     # Merge button
    #     if st.button("Merge Selected", key="merge_top"):
    #         # Store selected entries to merge in session state
    #         st.session_state["merging"] = selected.copy()

    #     # Merge form display immediately after the "Merge Selected" button is clicked
    #     if st.session_state.get("merging"):
    #         st.markdown("---")
    #         st.subheader("Merge Entries")
    #         merged_ids = st.session_state["merging"]
    #         st.info(f"Creating merged entry from: {merged_ids}")

    #         # Input for merged question and answer
    #         new_q = st.text_input("Merged Question", key="new_question")
    #         new_a = st.text_area("Merged Answer", key="new_answer")

    #         # Get categories from selected entries
    #         selected_categories = df[df["cqid"].isin(merged_ids)]["category"].dropna().unique().tolist()
    #         selected_categories.sort()

    #         st.markdown("### Category")
    #         selected_cat = st.selectbox("Select Category", options=selected_categories + ["Other"], key="sel_cat")
    #         custom_cat = ""
    #         if selected_cat == "Other":
    #             custom_cat = st.text_input("Enter Custom Category", key="custom_cat")

    #         # Use selected or custom category in the merged entry
    #         final_category = custom_cat if selected_cat == "Other" else selected_cat


    #         # Optionally, you could add any extra fields like details
    #         new_details = st.text_area(
    #             "Additional Details (Optional)", key="new_details"
    #         )

    #         if st.button("Confirm Merge", key="confirm_merge"):
    #             if not new_q or not new_a:
    #                 st.error("Both merged question and answer are required!")
    #             else:
    #                 # Create a new merged entry with the necessary details
    #                 new_entry = {
    #                     "cqid": str(
    #                         uuid.uuid4()
    #                     ),  # Generate a new UUID for the merged entry
    #                     "question": new_q,
    #                     "answer": new_a,
    #                     "category": final_category,
    #                     "duplicate_count": sum(
    #                         df[df["cqid"].isin(merged_ids)]["duplicate_count"]
    #                     ),  # Sum of duplicate counts
    #                     "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #                     "is_outdated": False,  # Newly created, so not outdated
    #                     "merged_ids": merged_ids,  # Store the original CQIDs that were merged
    #                     "details": new_details,  # Add any optional additional details
    #                 }

    #                 # Append the new merged entry to the session state
    #                 if "merged_entries" not in st.session_state:
    #                     st.session_state["merged_entries"] = []
    #                 st.session_state["merged_entries"].append(new_entry)

    #                 # Success message
    #                 st.success(
    #                     f"Merged {len(merged_ids)} entries into a new entry: {new_entry['cqid']}"
    #                 )

    #                 # Clear the merging session state to reset the merge process
    #                 st.session_state["merging"] = None

    #     st.markdown("---")

    #     # Rows with Flag/Archive actions
    #     for _, row in paged.iterrows():
    #         cols = st.columns([4, 1, 1])
    #         cols[0].markdown(f"{truncate(row['question'])}")
    #         if cols[1].button("Flag", key=f"flag_{row['cqid']}"):
    #             with st.expander(f"Flagged {row['cqid']} - Details", expanded=True):
    #                 st.write(
    #                     {
    #                         "Created At": row["created_at"],
    #                         "Question": row["question"],
    #                         "Answer": row.get("answer", ""),
    #                     }
    #                 )
    #         if cols[2].button("Archive", key=f"arch_{row['cqid']}"):
    #             with st.expander(
    #                 f"Archived {row['cqid']} on {datetime.now().date()}", expanded=True
    #             ):
    #                 st.write(
    #                     {
    #                         "Created At": row["created_at"],
    #                         "Question": row["question"],
    #                         "Answer": row.get("answer", ""),
    #                     }
    #                 )


# ---> Working code


    # with tabs[1]:  # Clusters with suggestions
    #     st.subheader("🔍 Clusters")

    #     if suggestion_df.empty:
    #         st.warning("No suggestions available for clusters.")
    #     else:
    #         # Filter by action type
    #         actions = suggestion_df['action'].unique().tolist()
    #         selected_action = st.selectbox("Filter by Action", ["All"] + actions, key="cluster_action")

    #         filtered = suggestion_df if selected_action == "All" else suggestion_df[suggestion_df['action'] == selected_action]

    #         # Filter by Cluster ID
    #         cluster_ids = sorted(filtered['cluster_id'].dropna().unique())
    #         selected_cluster = st.selectbox("Filter by Cluster ID", ["All"] + list(map(str, cluster_ids)), key="cluster_id_filter")

    #         if selected_cluster != "All":
    #             filtered = filtered[filtered['cluster_id'].astype(str) == selected_cluster]

    #         st.write(f"{len(filtered)} suggestions found for selected filter.")
    #         paged_filtered = paginate_df(filtered, key="suggestion_pagination")

    #         for i, row in paged_filtered.iterrows():
    #             cqid = row['cqid']
    #             action = row['action']
    #             target = row.get('target_cqid', None)
    #             cluster = row.get('cluster_id', 'N/A')
    #             key_suffix = f"{cqid}_{action}_{i}"

    #             # Avoid logging the same action twice
    #             if key_suffix in st.session_state["logged_actions"]:
    #                 continue

    #             qna_row = df[df['cqid'] == cqid]
    #             if qna_row.empty:
    #                 continue
    #             qna_row = qna_row.squeeze()

    #             with st.expander(f"📌 {action.upper()} — CQID: {cqid} | Cluster: {cluster}"):
    #                 st.markdown(f"**Product:** {qna_row.get('product_name', 'N/A')}")
    #                 st.markdown(f"**Category:** {qna_row.get('category', 'N/A')}")
    #                 st.markdown(f"**Question:** {qna_row.get('question', '')}")
    #                 st.markdown(f"**Details:** {qna_row.get('details', '')}")
    #                 st.markdown(f"**Answer:** {qna_row.get('answer', '')}")

    #                 if action in ["merge", "review"] and pd.notna(target):
    #                     st.markdown("---")
    #                     st.markdown(f"**Target → CQID: {target}**")
    #                     target_row = df[df['cqid'] == target]
    #                     if not target_row.empty:
    #                         target_row = target_row.squeeze()
    #                         st.markdown(f"**Target Product:** {target_row.get('product_name', 'N/A')}")
    #                         st.markdown(f"**Target Category:** {target_row.get('category', 'N/A')}")
    #                         st.markdown(f"**Target Question:** {target_row.get('question', '')}")
    #                         st.markdown(f"**Target Details:** {target_row.get('details', '')}")
    #                         st.markdown(f"**Target Answer:** {target_row.get('answer', '')}")
    #                     else:
    #                         st.warning("Target CQID not found in dataset.")

    #                 if action == "archive":
    #                     st.markdown("---")
    #                     st.markdown(f"**Created At:** {qna_row.get('created_at', '')}")
    #                     st.markdown(f"**Deleted At:** {qna_row.get('deleted_at', 'N/A')}")

    #                 col1, col2 = st.columns(2)
    #                 accept = col1.button("✅ Accept", key=f"accept_{key_suffix}")
    #                 reject = col2.button("❌ Reject", key=f"reject_{key_suffix}")

    #                 if accept or reject:
    #                     log_action(action, cqid, target)
    #                     st.session_state["logged_actions"].add(key_suffix)
    #                     st.success(f"{'Accepted' if accept else 'Rejected'} action for CQID {cqid}")


# ---> Working code


    with tabs[1]:  # Clusters with suggestions
        similarity_threshold = st.slider(
                "Select Similarity Threshold for Filtering or Manual Clustering",
                min_value=0.0, max_value=1.0, value=0.75, step=0.01
            )

        st.write(f"Current Threshold: **{similarity_threshold:.2f}**")
        
        st.subheader("🔍 Clusters")

        if suggestion_df.empty:
            st.warning("No suggestions available for clusters.")
        else:
            # Filter by action type
            actions = suggestion_df['action'].unique().tolist()
            selected_action = st.selectbox("Filter by Action", ["All"] + actions, key="cluster_action")

            filtered = suggestion_df if selected_action == "All" else suggestion_df[suggestion_df['action'] == selected_action]

            # Filter by Cluster ID
            cluster_ids = sorted(filtered['cluster_id'].dropna().unique())
            selected_cluster = st.selectbox("Filter by Cluster ID", ["All"] + list(map(str, cluster_ids)), key="cluster_id_filter")

            if selected_cluster != "All":
                filtered = filtered[filtered['cluster_id'].astype(str) == selected_cluster]

            st.write(f"{len(filtered)} suggestions found for selected filter.")
            paged_filtered = paginate_df(filtered, key="suggestion_pagination")

            for i, row in paged_filtered.iterrows():
                cqid = row['cqid']
                action = row['action']
                target = row.get('target_cqid', None)
                cluster = row.get('cluster_id', 'N/A')
                key_suffix = f"{cqid}_{action}_{i}"

                # Avoid logging the same action twice
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

                    # If action is "merge", show the merge form
                    if action == "merge" and pd.notna(target):
                        st.markdown("---")
                        st.subheader("Merge Entries")

                        # Input for merged question and answer
                        new_q = st.text_input("Merged Question", key=f"new_question_{cqid}")
                        new_a = st.text_area("Merged Answer", key=f"new_answer_{cqid}")

                        # Get categories from selected entries
                        selected_categories = df[df["cqid"].isin([cqid, target])]["category"].dropna().unique().tolist()
                        selected_categories.sort()

                        st.markdown("### Category")
                        selected_cat = st.selectbox("Select Category", options=selected_categories + ["Other"], key=f"sel_cat_{cqid}")
                        custom_cat = ""
                        if selected_cat == "Other":
                            custom_cat = st.text_input("Enter Custom Category", key=f"custom_cat_{cqid}")

                        # Use selected or custom category in the merged entry
                        final_category = custom_cat if selected_cat == "Other" else selected_cat

                        # Optionally, you could add any extra fields like details
                        new_details = st.text_area("Additional Details (Optional)", key=f"new_details_{cqid}")

                        if st.button("Confirm Merge", key=f"confirm_merge_{cqid}"):
                            if not new_q or not new_a:
                                st.error("Both merged question and answer are required!")
                            else:
                                # Create a new merged entry with the necessary details
                                new_entry = {
                                    "cqid": str(uuid.uuid4()),  # Generate a new UUID for the merged entry
                                    "question": new_q,
                                    "answer": new_a,
                                    "category": final_category,
                                    "duplicate_count": 1,  # Assuming merged entries have a count of 1
                                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "is_outdated": False,  # Newly created, so not outdated
                                    "merged_ids": [cqid, target],  # Store the original CQIDs that were merged
                                    "details": new_details,  # Add any optional additional details
                                }

                                print("New entry", new_entry)

                                # Log the new merged entry in the activity log (you can modify this based on your log format)
                                log_action("merge", new_entry,[cqid, target])

                                # Append the new merged entry to the session state
                                if "merged_entries" not in st.session_state:
                                    st.session_state["merged_entries"] = []
                                st.session_state["merged_entries"].append(new_entry)

                                # Success message
                                st.success(
                                    f"Merged {len([cqid, target])} entries into a new entry: {new_entry['cqid']}")
                                


    with tabs[2]:  # Outdated
        st.subheader("🆕 Outdated Q&A")
        old = filt[filt["is_outdated"]]
        paged = sanitize_df(paginate_df(old, "outdated"))
        st.dataframe(
            paged[["cqid", "question", "category", "created_at"]],
            use_container_width=True,
        )

    with tabs[3]:  # Compare
        st.subheader("🔁 Compare Entries")
        ids = list(df["cqid"].unique())
        a, b = st.columns(2)
        id1 = a.selectbox("First CQID", ids, key="c1")
        id2 = b.selectbox("Second CQID", ids, key="c2")
        r1 = df[df["cqid"] == id1].iloc[0]
        r2 = df[df["cqid"] == id2].iloc[0]
        col1, col2 = st.columns(2)
        col1.metric("CQID 1", id1)
        col2.metric("CQID 2", id2)
        for container, row in zip([col1, col2], [r1, r2]):
            container.write(
                row[
                    [
                        "product_name",
                        "category",
                        "question",
                        "answer",
                        "duplicate_count",
                        "created_at",
                    ]
                ]
            )

    with tabs[4]:  # Product Analysis
        st.subheader("📈 Product Analysis")
        prod_list = ["All"] + sorted(df["product_name"].unique())
        selected = st.selectbox("Select Product", prod_list)
        sub = df.copy() if selected == "All" else df[df["product_name"] == selected]
        # Metrics
        metrics = {
            "Total Q": len(sub),
            "Outdated %": f"{100*sub['is_outdated'].mean():.1f}%",
            "Avg Dup": f"{sub['duplicate_count'].mean():.1f}",
        }
        cols = st.columns(len(metrics))
        for c, (k, v) in zip(cols, metrics.items()):
            c.metric(k, v)
        # Time series
        ts = sub.groupby(sub["created_at"].dt.date).size().reset_index(name="count")
        fig_ts = px.line(
            ts, x="created_at", y="count", title="Questions Over Time", height=350
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Integrated grouped bar across products/categories
        st.markdown("---")
        st.markdown("### 🌈 Product vs. Category Distribution")
        gb = sub.groupby(["product_name", "category"]).size().reset_index(name="count")
        fig_grp = px.bar(
            gb,
            x="product_name",
            y="count",
            color="category",
            barmode="group",
            title="Questions by Product and Category",
            height=400,
        )
        st.plotly_chart(fig_grp, use_container_width=True)

    with tabs[5]:  # Category Matrix
        st.subheader("🔢 Category-Product Heatmap")
        pivot = filt.groupby(["product_name", "category"]).size().unstack(fill_value=0)
        fig_cm = px.imshow(
            pivot,
            labels={"x": "Category", "y": "Product", "color": "Count"},
            title="Questions per Product-Category",
            aspect="auto",
        )
        fig_cm.update_layout(margin=dict(l=100, r=20, t=50, b=100))
        st.plotly_chart(fig_cm, use_container_width=True)

    with tabs[6]:  # Time Trends
        st.subheader("📅 Time Trend: Outdated vs Current")
        st.markdown("Cumulative count of outdated vs. active Q&As over time.")
        fig6 = px.bar(
            time_series.reset_index(),
            x="date_only",
            y=["Cumulative_Outdated", "Cumulative_Current"],
            title="Cumulative Outdated vs Active Q&A",
            labels={"variable": "Status", "value": "Count", "date_only": "Date"},
        )
        st.plotly_chart(fig6, use_container_width=True)

        fig7 = px.line(
            time_series.reset_index(),
            x="date_only",
            y=["Cumulative_Outdated", "Cumulative_Current"],
            title="Trend of Outdated vs Active Q&A",
            labels={"variable": "Status", "value": "Count", "date_only": "Date"},
        )
        st.plotly_chart(fig7, use_container_width=True)

    with tabs[7]:  # 📜 Action Logs inside tab
        st.subheader("📋 Action Log History")

        # Load raw logs
        log_path = "data/action_log.csv"
        if os.path.exists(log_path):
            logs = pd.read_csv(log_path)
        else:
            st.warning("No actions logged yet.")
            st.stop()

        # Dedupe
        logs = logs.drop_duplicates(subset=["timestamp", "action", "cqid", "target_cqid"])
        if logs.empty:
            st.warning("No actions logged after dedupe.")
            st.stop()

        # Enrich with metadata
        lib_meta = (
            df.drop_duplicates(subset="cqid")
            .set_index("cqid")
            [["product_name", "category", "created_at", "deleted_at", "cluster_id"]]
        )
        logs = logs.merge(lib_meta, left_on="cqid", right_index=True, how="left")

        # Enrich with suggestion context
        sug_meta = (
            suggestion_df.drop_duplicates(subset="cqid")
            .set_index("cqid")
            [["target_cqid", "cluster_id"]]
            .rename(columns={
                "cluster_id": "suggested_cluster",
                "target_cqid": "merge_target_cqid"
            })
        )
        logs = logs.merge(sug_meta, left_on="cqid", right_index=True, how="left")

        # Display charts
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

        # Full log view
        st.markdown("### 🔎 Full Action Log")
        st.dataframe(logs.sort_values(by="timestamp", ascending=False), use_container_width=True)

        # Download button
        csv = logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Enriched Action Log",
            data=csv,
            file_name="action_log_enriched.csv",
            mime="text/csv"
        )

    with tabs[8]:  # All Entries
            st.subheader("📋 All Entries")
            st.dataframe(filt, use_container_width=True, height=350)

            st.markdown("### 🔍 View Full Entry Details")
            cqid_options = filt["cqid"].unique().tolist()
            selected_cqid = st.selectbox(
                "Start typing CQID",
                options=cqid_options,
                index=None,  # No default selection
                placeholder="Type or select a CQID",
                key="all_entries_select_cqid",
            )

            if selected_cqid:
                selected_row = filt[filt["cqid"] == selected_cqid].iloc[0].to_dict()
                st.json(selected_row)


    # Return to file selection
    if st.sidebar.button("← Change File"):
        st.session_state.df = None
        st.rerun()
