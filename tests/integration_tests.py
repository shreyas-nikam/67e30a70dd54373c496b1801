import unittest
import importlib

class TestMultiPageApp(unittest.TestCase):
    def test_page_navigation(self):
        # Check that the pages can be imported and have an 'app' callable.
        attack_viz = importlib.import_module("pages.attack_visualizer")
        data_overview = importlib.import_module("pages.data_overview")
        self.assertTrue(hasattr(attack_viz, "app"))
        self.assertTrue(hasattr(data_overview, "app"))
        
if __name__ == '__main__':
    unittest.main()
