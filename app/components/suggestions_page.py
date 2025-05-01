import streamlit as st
import pandas as pd
import os

DECISIONS_FILE = "data/curation_decisions.csv"

# Function to log decisions
def log_decision(df, cqid, action, decision):
    timestamp = pd.Timestamp.now().isoformat()
    new_row = {"cqid": cqid, "action": action, "decision": decision, "timestamp": timestamp}
    df = df.append(new_row, ignore_index=True)
    df.to_csv(DECISIONS_FILE, index=False)
    st.success(f"{decision.title()}ed suggestion for {cqid}")
    return df

def suggestions_page(library_df, suggestion_df):
    st.subheader("💡 Actionable Suggestions for Curator")
    st.markdown("Each suggestion is grouped by **action** and **cluster**. Accept or reject below.")

    # Initialize or load decision log
    if os.path.exists(DECISIONS_FILE):
        decision_log = pd.read_csv(DECISIONS_FILE)
    else:
        decision_log = pd.DataFrame(columns=["cqid", "action", "decision", "timestamp"])

    # Tabs by action
    action_tabs = st.tabs(["🔀 Merge", "🕵️ Review", "🗃️ Archive"])

    for action, tab in zip(["merge", "review", "archive"], action_tabs):
        with tab:
            act_df = suggestion_df[suggestion_df["action"] == action]
            if act_df.empty:
                st.info(f"No {action.title()} suggestions.")
                continue

            clusters = sorted(act_df['cluster_id'].dropna().unique())
            for cluster_id in clusters:
                cluster_sugs = act_df[act_df['cluster_id'] == cluster_id]
                cluster_entries = library_df[library_df['cluster_id'] == cluster_id]

                with st.expander(f"📎 Cluster {cluster_id} – {len(cluster_entries)} Q&As", expanded=False):
                    for _, row in cluster_sugs.iterrows():
                        cqid = row['cqid']
                        entry = library_df[library_df['cqid'] == cqid].iloc[0]

                        st.markdown(f"#### ❓ {entry['question']}")
                        st.markdown(f"**Answer:** {entry['answer']}")
                        st.markdown(f"**Product:** {entry['product_name']} | **Category:** {entry['category']}")
                        st.markdown(f"**Details:** {entry.get('details', '')}")

                        if action == "archive":
                            st.markdown(f"🗓️ Created: `{entry['created_at']}` | Deleted: `{entry['deleted_at']}`")

                        col1, col2 = st.columns(2)
                        key_accept = f"accept_{cqid}_{action}_{cluster_id}"
                        key_reject = f"reject_{cqid}_{action}_{cluster_id}"

                        if col1.button("👍 Accept", key=key_accept):
                            decision_log = log_decision(decision_log, cqid, action, "accept")

                        if col2.button("❌ Reject", key=key_reject):
                            decision_log = log_decision(decision_log, cqid, action, "reject")

    # Export option
    st.markdown("---")
    st.markdown("### 📤 Export Curation Decisions")
    if not decision_log.empty:
        st.dataframe(decision_log.sort_values(by='timestamp', ascending=False))
        csv = decision_log.to_csv(index=False).encode('utf-8')
        st.download_button("Download Decisions CSV", data=csv, file_name="curation_decisions.csv", mime='text/csv')
    else:
        st.info("No decisions logged yet.")
