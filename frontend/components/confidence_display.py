import streamlit as st

def show_confidence(data):
    st.subheader("Confidence")
    score = float(data.get("confidence_score", 0.0))
    score = max(0.0, min(1.0, score))
    st.progress(score)
    st.caption(f"confidence_score: {score:.2f}")
