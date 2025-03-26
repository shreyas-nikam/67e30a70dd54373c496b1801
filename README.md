# Adversarial Attack Visualizer

## Overview
This repository contains a Streamlit application showcasing the concept of adversarial machine learning attacks and their effects on a synthetic dataset. The application leverages:
- [SECML](https://github.com/secml/secml)
- [Foolbox](https://github.com/bethgelab/foolbox)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
  
Users can visualize and experiment with adversarial attack parameters, observing real-time impacts on model outputs through interactive charts. This project highlights Sections 2 and 3 of the reference document (on adversarial examples and attack taxonomies).

## Features
- Multi-page Streamlit application.
- Synthetic dataset loading and exploration.
- Interactive adversarial attack configuration using SECML, Foolbox, or ART.
- Real-time visualizations with Plotly.
- Thorough markdown explanations and tooltips describing each visualization and concept.

## Getting Started

### 1. Local Installation
1. Clone this repository.
2. Install Python 3.12 or higher.
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

### 2. Docker Usage
1. Clone this repository.
2. Build the Docker image:
   ```bash
   docker build -t adv-attack-visualizer .
   ```
3. Run the Docker container:
   ```bash
   docker run -p 8501:8501 adv-attack-visualizer
   ```
4. Access the application at http://localhost:8501 in your web browser.

## Project Structure
- **app.py**: Main Streamlit entry point (includes the boilerplate code).
- **pages/**: Directory containing multiple Streamlit pages (data overview, attack demos, etc.).
- **requirements.txt**: Python dependencies.
- **Dockerfile**: Docker build instructions.
- **README.md**: This file, explaining usage and features.

## Testing
- Test files (e.g., basic Python scripts) validate dataset loading, model training, and adversarial attack generation.

## Security and Code Validation
No destructive or suspicious commands (e.g., handling system files) exist in this repository. Any known vulnerabilities or red flags will be documented in the Streamlit frontend or addressed immediately.

## License & Acknowledgments
© 2025 QuantUniversity. All Rights Reserved.
The purpose of this demonstration is solely for educational use and illustration.
For more details, see the legal documentation link in the application interface.

