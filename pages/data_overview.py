import streamlit as st

def app():
    st.header("Data Overview & Lab Background")
    st.markdown(
        "## NIST AI - Adversarial Machine Learning Lab\n"
        "Welcome to this demonstration lab for adversarial machine learning. This lab leverages open-source tools including SECML, Foolbox, and the Adversarial Robustness Toolbox (ART) to help users understand how adversarial attacks can compromise machine learning models.\n\n"
        "### Learning Outcomes\n"
        "- **Insight**: Understand the vulnerabilities of models under adversarial attack.\n"
        "- **Conceptual Understanding**: Learn about adversarial examples and attack taxonomies.\n"
        "- **Practical Skills**: Experiment with parameters and visualize the effects of adversarial perturbations.\n\n"
        "### Dataset Details\n"
        "We use a synthetic dataset comprised of numeric values and simulated time series. This dataset is used solely for demonstrating data handling and interactive visualization.\n\n"
        "### Instructions\n"
        "Use the sidebar to navigate between pages. On the 'Adversarial Attack Visualizer' page, adjust the parameters to see how adversarial attacks may affect the outcome on a sample dataset."
    )
