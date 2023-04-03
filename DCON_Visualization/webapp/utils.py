import os

from config import Config

def create_data_folder(config: Config):
    if not os.path.exists(config.DATA_PATH):
        os.makedirs(config.DATA_PATH)

def check_loaded_datasets(config: Config) -> list:

    num_of_datasets = len(config.DATASET_URLS)
    loaded_datasets = []

    for ind in range(num_of_datasets):

        file_name = str(config.DATA_PATH)+"\Dataset"+str(ind+1)+"\dataframe.pkl"

        if os.path.exists(file_name):
            loaded_datasets.append(True)
        else:
            loaded_datasets.append(False)

    return loaded_datasets