from pathlib import Path

class Config:
    DATA_PATH = Path(__file__).parents[0]/'data'
    RANDOM_STATE = 12345
    DATASET_NAMES = [
        'Computer Hardware Dataset',
        'Forest Fires Dataset',
        'Stock Portfolio Performance Dataset',
        'Yacht Hydrodynamics Dataset',
        'Facebook Metrics Dataset',
        'Residential Building Dataset',
        'Real Estate Valuation Dataset',
        'QSAR Fish Toxicity Dataset',
        'QSAR Aquatic Toxicity Dataset'
    ]
    DATASET_URLS = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00390/stock%20portfolio%20performance%20data%20set.xlsx',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00505/qsar_aquatic_toxicity.csv'
    ]