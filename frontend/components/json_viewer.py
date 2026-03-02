import streamlit as st
import json

def show_json(data):
    st.subheader("Extracted Data")
    st.json(data)