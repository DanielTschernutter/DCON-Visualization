import unittest

from config import Config
from webapp.loader import load_dataset_from_UCL

class TestLoader(unittest.TestCase):
    
    def test_loading_datasets(self):
        config = Config()
        # Dataset 1
        output = load_dataset_from_UCL(config,0)
        self.assertTrue(output,"Failed to load dataset 0.")
        # Dataset 2
        output = load_dataset_from_UCL(config,1)
        self.assertTrue(output,"Failed to load dataset 1.")
        # Dataset 3
        output = load_dataset_from_UCL(config,2)
        self.assertTrue(output,"Failed to load dataset 2.")
        # Dataset 4
        output = load_dataset_from_UCL(config,3)
        self.assertTrue(output,"Failed to load dataset 3.")
        # Dataset 5
        output = load_dataset_from_UCL(config,4)
        self.assertTrue(output,"Failed to load dataset 4.")
        # Dataset 6
        output = load_dataset_from_UCL(config,5)
        self.assertTrue(output,"Failed to load dataset 5.")
        # Dataset 7
        output = load_dataset_from_UCL(config,6)
        self.assertTrue(output,"Failed to load dataset 6.")
        # Dataset 8
        output = load_dataset_from_UCL(config,7)
        self.assertTrue(output,"Failed to load dataset 7.")
        # Dataset 9
        output = load_dataset_from_UCL(config,8)
        self.assertTrue(output,"Failed to load dataset 8.")