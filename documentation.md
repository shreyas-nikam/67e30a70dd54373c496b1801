id: 67e30a70dd54373c496b1801_documentation
summary: NIST AI - Adversarial Machine Learning Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuCreate Streamlit Lab: Adversarial Attack Visualizer

This codelab will guide you through a Streamlit application designed to visualize and demonstrate adversarial attacks on machine learning models. This application is important because it highlights the vulnerability of even simple machine learning models to carefully crafted input perturbations. Understanding these vulnerabilities is crucial for developing more robust and secure AI systems.

This codelab covers the following concepts:

*   **Streamlit:** Building interactive web applications with Python.
*   **Adversarial Attacks:** Understanding the concept of adversarial examples and their impact on model predictions.
*   **Adversarial Robustness Toolbox (ART):** Using ART to generate adversarial examples.
*   **Fast Gradient Method (FGM):** Implementing a basic adversarial attack algorithm.
*   **Data Visualization:** Using Plotly to visualize data and attack results.

## Setting up the Environment
Duration: 00:05

Before you start, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install streamlit pandas numpy plotly scikit-learn adversarial-robustness-toolbox
```

Alternatively, you can run the included `test_run.py` script. This script primarily tests if all the required libraries are installed in your environment. To execute this script, run the command: `python test_run.py`

```python
import sys

def test_imports():
    try:
        import streamlit
        import plotly
        import sklearn
        import art
        import secml
        import foolbox
        print('All libraries imported successfully.')
    except ImportError as e:
        print(f'ImportError: {e}')
        sys.exit(1)

def main():
    test_imports()
    print('Basic test completed successfully.')

if __name__ == '__main__':
    main()
```

<aside class="positive">
It is good practice to set up a virtual environment to manage dependencies for each project to avoid conflicts.
</aside>

## Running the Application
Duration: 00:02

To run the Streamlit application, navigate to the directory containing `app.py` and execute the following command:

```bash
streamlit run app.py
```

This command will start the Streamlit server and open the application in your web browser.

## Exploring the Main Page (app.py)
Duration: 00:03

The `app.py` file serves as the main entry point for the Streamlit application. It configures the page layout, sets the title, and provides a brief introduction to the application.

```python
import streamlit as st

st.set_page_config(page_title="QuCreate Streamlit Lab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.write("Select a page from the sidebar to begin exploring the Adversarial Attack Visualizer.")

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
```

Key components:

*   `st.set_page_config()`: Sets the page title and layout.  `layout="wide"` utilizes the full screen width.
*   `st.sidebar.image()`: Adds an image to the sidebar, which in this case is the QuantUniversity logo.
*   `st.title()`: Displays the main title of the application.
*   `st.write()`: Writes text to the application.
*   `st.caption()`: Adds a caption at the bottom of the page, typically used for disclaimers.

This main page acts as a landing page, guiding the user to explore the other pages in the application using the sidebar.

## Understanding the Data Overview Page (pages/01_Data_Overview.py)
Duration: 00:10

This page provides an overview of a synthetic dataset used for demonstration purposes. It generates and visualizes the data using Pandas and Plotly.

```python
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
```

Here's a breakdown:

1.  **Data Generation**: A synthetic dataset is created using NumPy and Pandas. It includes numerical features (X, Y), a categorical feature (Category), and a time-series feature (Time).
2.  **Displaying the Data**: The first 10 rows of the dataset are displayed using `st.dataframe()`.
3.  **Visualizations**: Two interactive plots are generated using Plotly Express:
    *   A scatter plot of X vs. Y, colored by the Category.
    *   A line plot showing the time-series trend of Y.  `use_container_width=True` ensures that the plot scales to fit the container.
4.  **Key Observations**: Important characteristics of the data are highlighted using `st.markdown()`.

This page allows users to explore the dataset and understand its properties through interactive visualizations.  The synthetic data is designed to be simple and easily visualized, making it suitable for demonstrating adversarial attacks.

## Analyzing the Attack Demo Page (pages/02_Attack_Demo.py)
Duration: 00:20

This page demonstrates a simple adversarial evasion attack using the Adversarial Robustness Toolbox (ART). It shows how a small perturbation can significantly affect the model's predictions.

```python
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier

st.title("Adversarial Attack Demo")
st.write("### Demonstration of a Simple Adversarial Evasion Attack using ART")

st.markdown("**Overview**: This page showcases a minimal example of an adversarial attack on a simple logistic regression classifier. "
            "We use the Fast Gradient Method (FGM) from the Adversarial Robustness Toolbox (ART) library to generate adversarial samples. "
            "Adjust the perturbation level (epsilon) below to see how even slight changes can have a large impact on the model's predictions.")

# Generate synthetic 2D data
np.random.seed(42)
num_samples = 200
x1 = np.random.normal(loc=0, scale=1, size=num_samples)
x2 = x1 * 1.2 + np.random.normal(loc=0, scale=1, size=num_samples)
X = np.column_stack((x1, x2))
y = (x1 + x2 > 0).astype(int)  # Simple decision boundary

# Train a logistic regression classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Wrap the trained model in an ART SklearnClassifier
classifier = SklearnClassifier(model=model)

# User-specified epsilon for adversarial attack
epsilon = st.slider("FGM Epsilon (Attack Strength)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

# Instantiate and apply the Fast Gradient Method attack
attack = FastGradientMethod(estimator=classifier, eps=epsilon)
X_test_adv = attack.generate(x=X_test)

# Evaluate model on both clean and adversarial samples
clean_preds = model.predict(X_test)
adv_preds = model.predict(X_test_adv)

accuracy_clean = (clean_preds == y_test).mean()
accuracy_adv = (adv_preds == y_test).mean()

st.write(f"**Accuracy on clean test set**: {accuracy_clean:.2f}")
st.write(f"**Accuracy on adversarial test set**: {accuracy_adv:.2f}")

st.markdown("Below is a 2D visualization of a subset of the clean and adversarial samples. "
            "Points are colored by the predicted class of the logistic regression model.")

subset_size = 50
X_vis = X_test[:subset_size]
X_vis_adv = X_test_adv[:subset_size]
pred_clean_vis = model.predict(X_vis)
pred_adv_vis = model.predict(X_vis_adv)

df_vis = {
    "X1": np.concatenate((X_vis[:, 0], X_vis_adv[:, 0])),
    "X2": np.concatenate((X_vis[:, 1], X_vis_adv[:, 1])),
    "Type": ["Clean"] * subset_size + ["Adversarial"] * subset_size,
    "Predicted Class": np.concatenate((pred_clean_vis, pred_adv_vis)),
}

fig = px.scatter(df_vis,
                 x="X1",
                 y="X2",
                 color="Predicted Class",
                 symbol="Type",
                 title="Clean vs. Adversarial Samples (subset)",
                 labels={"X1": "Feature 1", "X2": "Feature 2"})
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Key Takeaway**: Notice how a small perturbation (controlled by epsilon) can drastically change the model's predictions, "
            "underscoring the vulnerability of machine learning models to adversarial attacks.")
```

The process can be described using the following steps:

1.  **Data Generation and Model Training**: Synthetic 2D data is generated, and a logistic regression model is trained on it. The data is split into training and testing sets.
2.  **ART Integration**: The trained model is wrapped in an ART `SklearnClassifier`.  This allows ART to interact with the model for attack generation and defense.
3.  **Adversarial Attack**: The Fast Gradient Method (FGM) attack is instantiated with a user-controlled epsilon value (perturbation strength). The `st.slider` widget allows the user to adjust the epsilon value interactively.  The attack is then applied to the test data to generate adversarial examples.
4.  **Evaluation**: The model is evaluated on both clean and adversarial examples, and the accuracy is displayed.
5.  **Visualization**: A scatter plot visualizes a subset of the clean and adversarial samples, colored by their predicted class. The `symbol` parameter distinguishes between clean and adversarial samples.
6.  **Key Findings**: Emphasizes the impact of small perturbations on the model's predictions.

<aside class="negative">
The epsilon parameter controls the magnitude of the perturbation. Higher epsilon values result in stronger attacks, but also more noticeable changes to the input data.
</aside>

### Fast Gradient Method (FGM)

FGM is a simple yet effective adversarial attack. It works by calculating the gradient of the loss function with respect to the input and then perturbing the input in the direction that maximizes the loss.

Mathematically, the adversarial example x' is generated as follows:

x' = x + epsilon * sign(grad(J(theta, x, y)))

where:

*   x is the original input
*   epsilon is the perturbation magnitude
*   J is the loss function
*   theta represents the model's parameters
*   y is the true label
*   grad is the gradient operator
*   sign is the sign function

### Architecture Diagram
```mermaid
graph LR
    A[Original Input (x)] --> B(Calculate Gradient);
    B --> C{Sign(Gradient)};
    C --> D[Epsilon (ε)];
    D --> E(ε * Sign(Gradient));
    E --> F{x' = x + E};
    F --> G[Adversarial Example (x')];
```

### Key Functionalities and Code Explanation:

*   **Import Necessary Libraries:**  The script imports libraries like `streamlit`, `numpy`, `plotly`, `sklearn`, and `art`.
*   **Generate Synthetic Data:** Creates a 2D synthetic dataset.
*   **Train Logistic Regression Model:** Trains a simple logistic regression model using scikit-learn.
*   **Wrap Model with ART:** Wraps the scikit-learn model with `SklearnClassifier` from the `art` library, which is necessary for using ART's attack methods.
*   **Implement FGM Attack:** Uses the `FastGradientMethod` from ART to generate adversarial examples.  The strength of the attack is controlled by the `epsilon` parameter, which is set using a Streamlit slider.
*   **Evaluate Model Performance:** Evaluates the model's accuracy on both the original test data and the adversarial examples.
*   **Visualize Results:**  Uses Plotly to display the original and adversarial examples, allowing users to visually assess the impact of the attack.

This page clearly demonstrates the vulnerability of machine learning models to adversarial attacks and provides a hands-on experience in generating and visualizing adversarial examples.

## Conclusion
Duration: 00:05

This codelab provided a step-by-step guide to understanding and running a Streamlit application that visualizes adversarial attacks. By exploring the Data Overview and Attack Demo pages, you learned how to generate synthetic data, train a simple machine learning model, and use the Adversarial Robustness Toolbox (ART) to create and visualize adversarial examples. This application serves as a valuable tool for understanding the vulnerabilities of machine learning models and the importance of developing robust defenses against adversarial attacks.
