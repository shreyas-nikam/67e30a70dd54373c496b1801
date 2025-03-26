import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Data Overview")
st.write("### Synthetic Dataset Exploration")

# Generate a small synthetic dataset for demonstration
np.random.seed(42)
x = np.random.normal(loc=0, scale=1, size=100)
y = x * 3.5 + np.random.normal(loc=0, scale=2, size=100)
category = np.where(x > 0, "PositiveX", "NegativeX")
time = pd.date_range("2023-01-01", periods=100, freq="D")

df = pd.DataFrame({
    "X": x,
    "Y": y,
    "Category": category,
    "Time": time
})

st.markdown("Below is a synthetic dataset with numeric (X, Y), categorical (Category), and time-series (Time) features.")
st.dataframe(df.head(10))

st.write("## Visualizations")
# Scatter plot
fig_scatter = px.scatter(df, x="X", y="Y", color="Category",
                         title="Scatter Plot of Synthetic Dataset",
                         labels={"X": "X value", "Y": "Y value"})
st.plotly_chart(fig_scatter, use_container_width=True)

# Line plot over time
fig_line = px.line(df, x="Time", y="Y",
                   title="Time-series Trend of Y",
                   labels={"Time": "Time", "Y": "Y value"})
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("**Key Observations**: - The synthetic data is randomly generated but demonstrates a linear trend plus noise between X and Y. - The categorical feature is determined by whether X is positive or negative. - The time-series aspect can be used to demonstrate adversarial perturbations in a time dimension if desired.")
