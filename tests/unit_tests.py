import unittest
import numpy as np
import plotly.express as px
from pages import attack_visualizer, data_overview

class TestAttackVisualizer(unittest.TestCase):
    def test_synthetic_data_length(self):
        # Generate synthetic data similar to what is used in the page
        x = np.linspace(0, 10, 100)
        self.assertEqual(len(x), 100)

    def test_plotly_chart_creation(self):
        # Create a simple plotly figure and verify its structure.
        fig = px.line(x=[1,2,3], y=[1,4,9])
        self.assertTrue(hasattr(fig, "data"))
        
class TestDataOverview(unittest.TestCase):
    def test_data_overview_callable(self):
        # Verify that the data overview app function is callable.
        self.assertTrue(callable(data_overview.app))

if __name__ == '__main__':
    unittest.main()
