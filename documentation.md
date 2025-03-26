id: 67e30a70dd54373c496b1801_documentation
summary: NIST AI - Adversarial Machine Learning Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuCreate Streamlit Lab Codelab: Exploring Adversarial Attacks and Data Visualization

This codelab provides a comprehensive guide to understanding and exploring the functionalities of the QuCreate Streamlit Lab application. This application serves as an educational tool to demonstrate adversarial machine learning concepts and data visualization techniques. You will learn how small perturbations, known as adversarial attacks, can affect machine learning model predictions, and how to visualize such effects using interactive plots. By the end of this codelab, you'll understand the application's structure, how to navigate between different pages, and how to interact with the adversarial attack visualizer.

## Setting Up the Environment
Duration: 00:05

Before diving into the application, ensure you have the following:

1.  **Python:** Python 3.6 or higher is required.
2.  **Streamlit:** Install Streamlit using pip:

    ```bash
    pip install streamlit
    ```
3.  **Plotly:** Install Plotly using pip:

    ```bash
    pip install plotly
    ```
4.  **Numpy:** Install Numpy using pip:

    ```bash
    pip install numpy
    ```

## Running the Application
Duration: 00:02

1.  Save the provided code into the following files:

    *   `app.py` (main application file)
    *   `pages/attack_visualizer.py`
    *   `pages/data_overview.py`
    *   `tests/unit_tests.py`
    *   `tests/integration_tests.py`

    Make sure to create the `pages` and `tests` directories.

2.  Navigate to the directory containing `app.py` in your terminal.
3.  Run the application using the following command:

    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.

## Understanding the Application Structure
Duration: 00:05

The application is structured as a multi-page Streamlit application. Here's a breakdown:

*   **`app.py`:** This is the main entry point of the application. It handles the overall layout, title, sidebar navigation, and dynamic page loading.
*   **`pages/attack_visualizer.py`:** This module contains the code for the "Adversarial Attack Visualizer" page, allowing users to simulate and visualize the effects of adversarial attacks on synthetic data.
*   **`pages/data_overview.py`:** This module provides an overview of the application's purpose, learning outcomes, dataset details, and instructions.
*   **`tests/unit_tests.py`:** This module contains unit tests for individual components of the application, like data generation and chart creation.
*   **`tests/integration_tests.py`:** This module contains integration tests to ensure different parts of the application work together correctly, like page navigation.

The `app.py` file dynamically imports the selected page module based on the user's selection in the sidebar. This allows for a modular and organized application structure.

## Exploring the Data Overview Page
Duration: 00:05

1.  In the Streamlit application, observe the sidebar on the left.
2.  Select "Data Overview" from the selectbox.

The "Data Overview" page provides context for the application. It explains the purpose of the lab, which is to demonstrate adversarial machine learning concepts using open-source tools. It also outlines the learning outcomes, dataset details, and instructions for using the application.  The page emphasizes the use of a synthetic dataset for demonstration purposes.

## Interacting with the Adversarial Attack Visualizer
Duration: 00:15

1.  In the Streamlit application, select "Adversarial Attack Visualizer" from the sidebar.

This page allows you to visualize the impact of adversarial attacks on a synthetic dataset.  The page generates a sine wave with added noise as the original signal. You can then introduce a perturbation, simulating an adversarial attack, and observe the change in the signal.

2.  **Adjusting the Perturbation Magnitude:**

    *   Locate the "Attack Parameters" section in the sidebar.
    *   Use the "Perturbation Magnitude" slider to adjust the noise level.  The slider ranges from 0.0 to 1.0.
    *   Observe how the "Perturbed Signal (Adversarial Attack)" graph changes as you increase the perturbation magnitude.  Even small changes can significantly alter the signal.

3.  **Understanding the Graphs:**

    *   The "Original Signal" graph displays the synthetic sine wave with initial noise.
    *   The "Perturbed Signal (Adversarial Attack)" graph shows the signal after applying the adversarial perturbation. Notice how the signal deviates from the original as you increase the "Perturbation Magnitude".

<aside class="positive">
  The adversarial attack is simulated by adding random noise to the original signal. In real-world scenarios, adversarial attacks are carefully crafted perturbations designed to fool machine learning models.
</aside>

## Understanding the Code: `attack_visualizer.py`
Duration: 00:10

Let's examine the code behind the "Adversarial Attack Visualizer":

```python
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
```

*   **Import Statements:** The code imports `streamlit`, `numpy`, and `plotly.express`.
*   **`app()` function:** This function contains the logic for the page.
*   **Header and Markdown:**  The code displays a header and introductory text using `st.header()` and `st.markdown()`.
*   **Sidebar Slider:**  A slider is created in the sidebar using `st.sidebar.slider()` to control the "Perturbation Magnitude".
*   **Synthetic Data Generation:**  NumPy is used to generate a synthetic dataset (a sine wave with noise).
*   **Adversarial Perturbation:**  The adversarial perturbation is applied by adding random noise, scaled by the `noise_level`, to the original signal.
*   **Plotly Charts:**  Plotly Express is used to create line charts of the original and perturbed signals, which are then displayed using `st.plotly_chart()`.

## Running Tests
Duration: 00:05

To ensure the application is functioning correctly, you can run the unit and integration tests.

1.  Navigate to the directory containing the `tests` directory in your terminal.
2.  Run the unit tests:

    ```bash
    python -m unittest tests/unit_tests.py
    ```

3.  Run the integration tests:

    ```bash
    python -m unittest tests/integration_tests.py
    ```

The tests verify the basic functionality of the application, such as data generation, chart creation, and page navigation.

## Expanding the Application
Duration: 00:15

This application provides a foundation for exploring adversarial machine learning concepts. Here are some ideas for expanding the application:

*   **Implement different attack methods:** Instead of simply adding random noise, explore more sophisticated attack algorithms like the Fast Gradient Sign Method (FGSM).
*   **Use a real dataset:**  Integrate a real-world dataset and train a simple machine learning model. Then, demonstrate how adversarial attacks can affect the model's predictions.
*   **Add robustness techniques:**  Implement and visualize techniques for defending against adversarial attacks, such as adversarial training or input sanitization.
*   **Interactive visualizations:** Enhance the visualizations to provide more detailed insights into the effects of adversarial attacks. For example, you could display the difference between the original and perturbed signals.

<aside class="negative">
  Remember that adversarial machine learning is a constantly evolving field. It's important to stay up-to-date with the latest research and techniques.
</aside>

## Conclusion

This codelab provided a hands-on introduction to the QuCreate Streamlit Lab application. You learned how to navigate the application, interact with the adversarial attack visualizer, and understand the underlying code. By expanding upon this foundation, you can further explore the fascinating and important field of adversarial machine learning.
