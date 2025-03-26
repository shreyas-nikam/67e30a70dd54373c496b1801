id: 67e30a70dd54373c496b1801_user_guide
summary: NIST AI - Adversarial Machine Learning User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuCreate Streamlit Lab: Exploring Adversarial Attacks

This codelab will guide you through a Streamlit application designed to illustrate the fascinating and important concept of adversarial attacks in machine learning.  Machine learning models, despite their successes, are often vulnerable to subtle, carefully crafted perturbations in their inputs that can cause them to make incorrect predictions. This application provides a hands-on way to explore these vulnerabilities.  We'll be looking at synthetic data and a simple attack to understand the core ideas.

## Application Overview
Duration: 00:02

This Streamlit application consists of two main pages:

1.  **Data Overview:**  This page introduces a synthetic dataset used for demonstration purposes. It allows you to visualize the data and understand its characteristics. Understanding your data is a crucial first step in any machine learning task, especially when considering adversarial attacks.

2.  **Attack Demo:** This page showcases a simplified adversarial attack on a logistic regression model. You'll be able to adjust the strength of the attack and observe its effect on the model's accuracy and predictions. This is where you will directly interact with the concept of adversarial perturbations.

<aside class="positive">
This application is designed for educational purposes, giving you a visual and interactive understanding of adversarial attacks.
</aside>

## Navigating the Application
Duration: 00:01

The application uses a sidebar for navigation. On the left side of your screen, you will find a sidebar containing the title "QuLab" and a list of pages: "Data Overview" and "Attack Demo". Click on these links to navigate to the corresponding pages.

## Data Overview
Duration: 00:05

This page presents a synthetic dataset generated using NumPy and visualized using Plotly.

1.  **Dataset Generation:** The application generates a dataset with features named "X", "Y", "Category" and "Time". "X" and "Y" are numerical features related to each other with added noise. "Category" is a categorical feature depending on the value of "X" (PositiveX or NegativeX). "Time" is a time-series feature.
    
2.  **Dataframe Display:** The first few rows of the generated dataframe are displayed using `st.dataframe`, allowing you to inspect the data directly.

3.  **Visualizations:** Two interactive Plotly charts are displayed:
    *   A scatter plot of "X" vs. "Y", colored by the "Category" feature.  This helps visualize the relationship between the numerical and categorical features.
    *   A line plot of "Y" over "Time", showing a simple time series.

<aside class="positive">
Take some time to examine the dataframe and interact with the plots. Hover over the data points in the scatter plot to see individual data values. Understanding the data generation process will make the attack demo easier to understand.
</aside>

## Attack Demo
Duration: 00:10

This page demonstrates a basic adversarial attack using the Adversarial Robustness Toolbox (ART) library.

1.  **Model Training:** A logistic regression model is trained on the synthetic data to classify data points based on features X1 and X2. The dataset is split into training and test sets.

2.  **Adversarial Attack:** The Fast Gradient Method (FGM) is used to generate adversarial examples.  FGM calculates the gradient of the loss function with respect to the input and then adds a small perturbation in the direction of the gradient.  This perturbation is designed to fool the model.

3.  **Epsilon Slider:** A slider allows you to control the `epsilon` parameter of the FGM attack.  Epsilon determines the magnitude of the perturbation applied to the input data.  A larger epsilon means a stronger attack.

4.  **Accuracy Evaluation:** The model is evaluated on both the original (clean) test set and the generated adversarial test set. The accuracy scores are displayed using `st.write`.

5.  **Visualization:** A scatter plot visualizes a subset of the clean and adversarial samples. Points are colored by their *predicted* class.  Clean and adversarial samples are differentiated by shape.

<aside class="negative">
Experiment with the `epsilon` slider. Observe how increasing epsilon decreases the accuracy on the adversarial test set, and how this translates to data points changing class in the visualization.  Small changes to the input data can drastically affect the model's performance!
</aside>

## Understanding Adversarial Attacks
Duration: 00:02

This application demonstrates a key concept: machine learning models can be vulnerable to adversarial attacks. By carefully crafting small perturbations to the input data, it's possible to fool the model into making incorrect predictions. This has significant implications for the security and reliability of machine learning systems, especially in sensitive applications like autonomous driving, fraud detection, and medical diagnosis.

## Further Exploration
Duration: 00:05

This codelab provides a basic introduction to adversarial attacks. You can explore further by:

*   Trying different attack methods available in the ART library.
*   Experimenting with different model architectures.
*   Investigating techniques for defending against adversarial attacks (adversarial training, input sanitization, etc.).
*   Applying these concepts to real-world datasets and problems.

<aside class="positive">
The world of adversarial machine learning is constantly evolving. By understanding the basic principles and exploring the available tools, you can contribute to building more robust and secure machine learning systems.
</aside>
