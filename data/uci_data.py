from xml.etree.ElementInclude import include
import pandas as pd
from augmentation import augment_features
from typing import Tuple

DATASET_PATH = "data/heart_disease/heart_disease_uci.csv"
GENERATED_PATH = "data/heart_disease/heart_disease_generated.csv"

class UCIData():
    def __init__(self):
        self.dataset = pd.read_csv(DATASET_PATH)
        self.generated_data = pd.read_csv(GENERATED_PATH)
        print(f"Loaded UCI Heart Disease dataset with {self.dataset.shape[0]} rows and {self.dataset.shape[1]} columns.")
        print(f"Loaded generated data with {self.generated_data.shape[0]} rows and {self.generated_data.shape[1]} columns.")
    
    def get_data(self):
        return (self.dataset, self.generated_data)
    
    def get_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (self.dataset.drop(columns=['num']), self.generated_data.drop(columns=['num']))

    def get_targets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (self.dataset['num'], self.generated_data['num'])
    
    def get_headers(self) -> list:
        return self.dataset.columns.to_list()
    	
    def augment(self, num_aug=None):
        self.dataset = augment_features(self.dataset, num_aug=num_aug)
        self.generated_data = augment_features(self.generated_data, num_aug=num_aug)