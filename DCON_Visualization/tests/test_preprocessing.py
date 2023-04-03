import unittest
import os

from config import Config
from webapp.preprocessing import preprocess_dataset
from webapp.loader import load_dataset_from_UCL

def file_exists(config: Config, ind: int):
    path = str(config.DATA_PATH)+"\Dataset"+str(ind+1)+"\dataframe.pkl"
    return os.path.exists(path)

class TestPreprocessing(unittest.TestCase):
    
    def test_preprocessing_datasets(self):
        config = Config()
        # Dataset 1
        if file_exists(config,0):
            output = preprocess_dataset(config,0)
            self.assertIsNotNone(output,"Failed to preprocess dataset 0.")
        else:
            output = load_dataset_from_UCL(config,0)
            self.assertTrue(output,"Failed to load dataset 0.")
            output = preprocess_dataset(config,0)
            self.assertIsNotNone(output,"Failed to preprocess dataset 0.")

        # Dataset 2
        if file_exists(config,1):
            output = preprocess_dataset(config,1)
            self.assertIsNotNone(output,"Failed to preprocess dataset 1.")
        else:
            output = load_dataset_from_UCL(config,1)
            self.assertTrue(output,"Failed to load dataset 1.")
            output = preprocess_dataset(config,1)
            self.assertIsNotNone(output,"Failed to preprocess dataset 1.")

        # Dataset 3
        if file_exists(config,2):
            output = preprocess_dataset(config,2)
            self.assertIsNotNone(output,"Failed to preprocess dataset 2.")
        else:
            output = load_dataset_from_UCL(config,2)
            self.assertTrue(output,"Failed to load dataset 2.")
            output = preprocess_dataset(config,2)
            self.assertIsNotNone(output,"Failed to preprocess dataset 2.")

        # Dataset 4
        if file_exists(config,3):
            output = preprocess_dataset(config,3)
            self.assertIsNotNone(output,"Failed to preprocess dataset 3.")
        else:
            output = load_dataset_from_UCL(config,3)
            self.assertTrue(output,"Failed to load dataset 3.")
            output = preprocess_dataset(config,3)
            self.assertIsNotNone(output,"Failed to preprocess dataset 3.")

        # Dataset 5
        if file_exists(config,4):
            output = preprocess_dataset(config,4)
            self.assertIsNotNone(output,"Failed to preprocess dataset 4.")
        else:
            output = load_dataset_from_UCL(config,4)
            self.assertTrue(output,"Failed to load dataset 4.")
            output = preprocess_dataset(config,4)
            self.assertIsNotNone(output,"Failed to preprocess dataset 4.")

        # Dataset 6
        if file_exists(config,5):
            output = preprocess_dataset(config,5)
            self.assertIsNotNone(output,"Failed to preprocess dataset 5.")
        else:
            output = load_dataset_from_UCL(config,5)
            self.assertTrue(output,"Failed to load dataset 5.")
            output = preprocess_dataset(config,5)
            self.assertIsNotNone(output,"Failed to preprocess dataset 5.")

        # Dataset 7
        if file_exists(config,6):
            output = preprocess_dataset(config,6)
            self.assertIsNotNone(output,"Failed to preprocess dataset 6.")
        else:
            output = load_dataset_from_UCL(config,6)
            self.assertTrue(output,"Failed to load dataset 6.")
            output = preprocess_dataset(config,6)
            self.assertIsNotNone(output,"Failed to preprocess dataset 6.")

        # Dataset 8
        if file_exists(config,7):
            output = preprocess_dataset(config,7)
            self.assertIsNotNone(output,"Failed to preprocess dataset 7.")
        else:
            output = load_dataset_from_UCL(config,7)
            self.assertTrue(output,"Failed to load dataset 7.")
            output = preprocess_dataset(config,7)
            self.assertIsNotNone(output,"Failed to preprocess dataset 7.")

        # Dataset 9
        if file_exists(config,8):
            output = preprocess_dataset(config,8)
            self.assertIsNotNone(output,"Failed to preprocess dataset 8.")
        else:
            output = load_dataset_from_UCL(config,8)
            self.assertTrue(output,"Failed to load dataset 8.")
            output = preprocess_dataset(config,8)
            self.assertIsNotNone(output,"Failed to preprocess dataset 8.")