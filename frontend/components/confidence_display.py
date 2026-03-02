import streamlit as st

def show_confidence(data):
    st.subheader("Confidence Scores")
    for field, details in data.items():
        st.write(f"{field}: {details['confidence']}%")