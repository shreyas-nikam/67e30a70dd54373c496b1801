id: 67e30a70dd54373c496b1801_user_guide
summary: NIST AI - Adversarial Machine Learning User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuCreate Streamlit Lab: A User Guide

This codelab guides you through the QuCreate Streamlit Lab, an interactive tool designed to explore the concepts of adversarial machine learning. In this lab, you will learn about adversarial attacks, their impact on machine learning models, and ways to visualize these attacks. The lab uses a synthetic dataset to illustrate these concepts, allowing you to experiment with different attack parameters and observe their effects.

## Navigating the QuLab Interface
Duration: 00:02

The QuLab interface is designed for easy navigation. You will find:

*   **Sidebar**: The sidebar on the left provides access to different pages within the application. Use the selectbox to switch between the "Adversarial Attack Visualizer" and the "Data Overview" pages.
*   **Main Area**: This is where the content of the selected page is displayed, including explanations, interactive visualizations, and controls.
*   **Footer**: The footer contains copyright information and a disclaimer regarding the educational purpose of this demonstration.

## Understanding the Data Overview Page
Duration: 00:05

The "Data Overview" page serves as an introduction to the lab and the concepts it covers.

1.  **Select the Page:** Use the sidebar to select the "Data Overview" option.
2.  **Review the Content:** Read through the information provided. This page outlines the purpose of the lab, the learning outcomes, and details about the synthetic dataset used in the demonstrations.
3.  **Learning Outcomes:** Pay close attention to the learning outcomes to understand what you should gain from this lab. You'll learn about the vulnerabilities of machine learning models to adversarial attacks, gain a conceptual understanding of adversarial examples and attack taxonomies, and develop practical skills in experimenting with attack parameters.
4.  **Dataset Details:** Understand that the synthetic dataset is used for demonstration purposes, and the principles can be applied to real-world datasets.
<aside class="positive">
The Data Overview page provides essential context for the rest of the lab. Make sure you understand the information presented here before moving on to the next page.
</aside>

## Visualizing Adversarial Attacks
Duration: 00:10

The "Adversarial Attack Visualizer" page allows you to interactively explore the effects of adversarial attacks on a synthetic signal.

1.  **Select the Page:** Use the sidebar to select the "Adversarial Attack Visualizer" option.
2.  **Observe the Original Signal:** The first graph displays the original, unperturbed signal. This represents the "clean" data that a machine learning model would typically be trained on.
3.  **Adjust the Perturbation Magnitude:** The sidebar contains a slider labeled "Perturbation Magnitude". This slider controls the level of noise added to the original signal, simulating an adversarial attack.
4.  **Observe the Perturbed Signal:** The second graph displays the signal after the adversarial perturbation has been applied. As you adjust the "Perturbation Magnitude" slider, observe how the signal changes. Even small perturbations can significantly alter the signal.
5.  **Analyze the Impact:** Consider how these changes in the signal could affect the predictions of a machine learning model. In a real-world scenario, these perturbations could lead to misclassification or incorrect output.

<aside class="negative">
Keep the perturbation magnitude to a smaller value to observe the effect of small changes to the original signal.
</aside>

## Experimenting with Attack Parameters
Duration: 00:15

The "Adversarial Attack Visualizer" page is designed for experimentation.

1.  **Vary the Perturbation Magnitude:** Experiment with different values for the "Perturbation Magnitude" slider. Observe how the signal changes as you increase or decrease the magnitude of the perturbation.
2.  **Consider Real-World Implications:** Think about how these types of attacks could be used in real-world scenarios. For example, an attacker could add small amounts of noise to an image to cause a facial recognition system to misidentify the person in the image.
3.  **Reflect on Model Vulnerabilities:** This exercise highlights the vulnerabilities of machine learning models to adversarial attacks. Even models that perform well on clean data can be easily fooled by carefully crafted perturbations.

## Conclusion
Duration: 00:03

By completing this codelab, you should have a better understanding of adversarial attacks and their potential impact on machine learning models. You have learned how to visualize these attacks using the "Adversarial Attack Visualizer" and have experimented with different attack parameters. This knowledge is essential for developing more robust and secure machine learning systems.
