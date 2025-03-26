import streamlit as st
import numpy as np
import plotly.express as px

def app():
    st.header("Adversarial Attack Visualizer")
    st.markdown(
        "This page demonstrates the effect of adversarial attacks on a synthetic dataset. "
        "Adjust the parameters below to simulate how small perturbations can affect a machine learning model's predictions. "
        "For illustration purposes, we simulate an 'attack' by perturbing data."
    )
    
    st.sidebar.markdown("### Attack Parameters")
    noise_level = st.sidebar.slider("Perturbation Magnitude", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    # Synthetic data generation
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)
    
    # Apply adversarial perturbation
    y_adv = y + noise_level * np.random.normal(size=y.shape)
    
    fig = px.line(x=x, y=y, title="Original Signal", labels={"x": "X", "y": "Signal"})
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.line(x=x, y=y_adv, title="Perturbed Signal (Adversarial Attack)", labels={"x": "X", "y": "Signal"})
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown(
        "The above graphs illustrate how a small change in the input (perturbation magnitude) can affect the underlying signal. "
        "In a real adversarial attack scenario, such perturbations could lead to misclassification or significant deviations in model output."
    )
