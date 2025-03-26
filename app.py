import streamlit as st
import importlib

st.set_page_config(page_title="QuCreate Streamlit Lab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

# Multipage navigation using sidebar selectbox
pages = {
    "Adversarial Attack Visualizer": "pages.attack_visualizer",
    "Data Overview": "pages.data_overview"
}

page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Dynamically import and run the chosen page
selected_module = importlib.import_module(pages[page])
selected_module.app()

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. To access the full legal documentation, please visit this link. Any reproduction of this demonstration requires prior written consent from QuantUniversity.")
