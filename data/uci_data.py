import pandas as pd
from .augmentation import augment_features
from typing import Tuple, Any
import torch

DATASET_PATH = "data/heart_disease/heart_disease_uci.csv"
GENERATED_PATH = "data/heart_disease/heart_disease_generated.csv"

CONCEPT_PATH = "data/heart_disease/heart_disease_uci_concepts.csv"

class UCIDataset(torch.utils.data.Dataset):
    def __init__(self, generated: bool = False):
        self.dataset = pd.read_csv(DATASET_PATH)
        print(f"Loaded UCI Heart Disease dataset with {self.dataset.shape[0]} rows and {self.dataset.shape[1]} columns.")
        
        if generated:
            self.generated_data = pd.read_csv(GENERATED_PATH)
            self.dataset = pd.concat([self.dataset, self.generated_data], ignore_index=True)
            print(f"Loaded generated data with {self.generated_data.shape[0]} rows and {self.generated_data.shape[1]} columns.")

        # Split once so __getitem__ doesn’t recompute every time
        self.features = self.dataset.drop(columns=['num', 'id'])
        self.targets = self.dataset['num']

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return (features, target) pair at index idx."""
        x = self.features.iloc[idx].to_numpy()  # convert to numpy array for model use
        y = self.targets.iloc[idx]
        return x, y
    
    def get_data(self):
        return self.dataset
    
    def get_features(self) -> pd.DataFrame:
        return self.features

    def get_targets(self) -> pd.Series:
        return self.targets
    
    def get_headers(self) -> list:
        return self.dataset.columns.to_list()
    	
    def augment(self, num_aug=None):
        self.dataset = augment_features(self.dataset, num_aug=num_aug)
        # Refresh cached splits
        self.features = self.dataset.drop(columns=['num', 'id'])
        self.targets = self.dataset['num']

class ConceptDataset(torch.utils.data.Dataset):
    def __init__(self, generated: bool = False):
        self.dataset = pd.read_csv(CONCEPT_PATH)
        print(f"Loaded UCI Heart Disease Concept dataset with {self.dataset.shape[0]} rows and {self.dataset.shape[1]} columns.")
        
        if generated:
            self.generated_data = pd.read_csv(GENERATED_PATH)
            self.dataset = pd.concat([self.dataset, self.generated_data], ignore_index=True)
            print(f"Loaded generated data with {self.generated_data.shape[0]} rows and {self.generated_data.shape[1]} columns.")

        # Split once so __getitem__ doesn’t recompute every time
        self.features = self.dataset
        self.targets = self.dataset['label']

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return (features, target) pair at index idx."""
        x = self.features.iloc[idx].to_numpy()  # convert to numpy array for model use
        y = self.targets.iloc[idx]
        return x, y
    
    def get_data(self):
        return self.dataset
    
    def get_features(self) -> pd.DataFrame:
        return self.features

    def get_targets(self) -> pd.Series:
        return self.targets
    
    def get_headers(self) -> list:
        return self.dataset.columns.to_list()
