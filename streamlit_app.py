import streamlit as st
import eda, predict

with st.sidebar:
    st.title("Page Navigation")
    # input
    page = st.radio("Page", ("EDA", "Model Demo"))

    st.write("# About")
    st.write(
        """Page ini adalah informasi data calon nasabah"""
    )

if page == "EDA":
    eda.run()
else:
    predict.run()
