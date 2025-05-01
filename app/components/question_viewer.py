import streamlit as st

def display_questions(state_manager, category_filter):
    # Get the filtered data
    questions = state_manager.get_filtered_questions(category_filter)


    # Paginate in batches
    batch_size = 10
    total_batches = len(questions) // batch_size + 1
    batch_num = st.slider("Select Batch", min_value=1, max_value=total_batches, value=1)

    # Show batch of questions
    start_idx = (batch_num - 1) * batch_size
    end_idx = start_idx + batch_size
    batch_questions = questions[start_idx:end_idx]
    
    for question in batch_questions:
        print(question)
        st.subheader(f"Question: {question['question']}")
        st.text(f"Details: {question['details']}")
        st.text(f"Suggestion: {question['suggestion']}")
        st.text(f"Category: {question['category']}")
        st.text(f"Cluster ID: {question['cluster_id']}")
        break
