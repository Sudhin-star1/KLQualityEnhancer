import sys
sys.path.insert(0, '/Users/sudhinkarki/Desktop/Hackathon2/klqe/pipeline')

import streamlit as st
from logger import log_user_decision

def display_suggestions(state_manager):
    # Display suggestions for the current batch
    suggestions = state_manager.get_suggestions()
    
    for suggestion in suggestions:
        with st.expander(f"Suggestion for Question ID {suggestion['question_id']}"):
            st.write(f"Suggestion: {suggestion['suggestion']}")

            # Accept or Reject
            accept_button = st.button("Accept", key=f"accept_{suggestion['question_id']}")
            reject_button = st.button("Reject", key=f"reject_{suggestion['question_id']}")

            if accept_button:
                log_user_decision(suggestion['question_id'], suggestion['suggestion'], True)
                st.success(f"Suggestion accepted for Question ID {suggestion['question_id']}")
            elif reject_button:
                log_user_decision(suggestion['question_id'], suggestion['suggestion'], False)
                st.warning(f"Suggestion rejected for Question ID {suggestion['question_id']}")
