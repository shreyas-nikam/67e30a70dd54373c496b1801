# NIST AI - Adversarial Machine Learning Lab

This repository contains a Streamlit-based lab for demonstrating adversarial attacks in machine learning. The application is a multi-page interactive tool which showcases:
- Interactive adversarial attack visualizations using Plotly.
- Detailed documentation and explanations regarding the concepts behind adversarial machine learning.
- Integration of open-source libraries like SECML, Foolbox, and ART.

## Features

- **Multi-Page Application**: Seamlessly navigate between an interactive adversarial attack visualizer and data overview/documentation.
- **Interactive Visualizations**: Use sliders and controls to adjust attack parameters and see real-time Plotly graph updates.
- **Testing**: Includes both unit tests and integration tests to ensure reliability.
- **Dockerized**: Contains a Dockerfile and docker-compose.yml for easy containerization.

## Installation

1. Clone the repository.
2. Install dependencies using:
   ```
   pip install -r requirements.txt
   ```
3. To run the application:
   ```
   streamlit run app.py
   ```

## Running Tests

- Unit Tests:
  ```
  python -m unittest tests/unit_tests.py
  ```
- Integration Tests:
  ```
  python -m unittest tests/integration_tests.py
  ```

## Docker Instructions

To build and run using Docker:
1. Build the Docker image:
   ```
   docker build -t adversarial_lab .
   ```
2. Run the container:
   ```
   docker run -p 8501:8501 adversarial_lab
   ```

For docker-compose:

## License

© 2025 QuantUniversity. All Rights Reserved.
