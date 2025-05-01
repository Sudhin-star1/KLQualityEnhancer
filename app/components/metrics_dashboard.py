import streamlit as st

def display_metrics(state_manager):
    st.header("Knowledge Library Metrics")
    
    # Get and display stats
    stats = state_manager.get_metrics()
    
    for category, metric in stats.items():
        st.subheader(f"Category: {category}")
        st.write(f"Most Duplicates: {metric['most_duplicates']}")
        st.write(f"Most Outdated: {metric['most_outdated']}")
