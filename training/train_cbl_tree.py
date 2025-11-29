import argparse
import torch
import torch.nn.functional as F
import numpy as np
from modules import TCBL
import time
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data import *

CBL_PATH = "model_checkpoints/cbl/"
DIST = 0.3

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)


if __name__=="__main__":

    args = parser.parse_args()

    uci_ds = UCIDataset()
    uci_ds.augment()
    print("Augmented uci dataset: ", uci_ds.get_data().head())
    concept_ds = ConceptDataset()

    if len(uci_ds) != len(concept_ds):
        print(f"Dataset mismatch: UCIDataset={len(uci_ds)} vs ConceptDataset={len(concept_ds)}")
        exit(0)

    print("\nDatasets loaded successfully. Datasets have same number of rows.")

    # Convert entire dataset into tensors so normalization only happens **once**
    uci_features = np.vstack([uci_ds[i][0] for i in range(len(uci_ds))])
    uci_features = np.nan_to_num(uci_features, nan=0.0, posinf=1e6, neginf=-1e6)

    uci_labels   = np.vstack([concept_ds[i][0] for i in range(len(concept_ds))])

    print("uci features shape: ", uci_features.shape, " uci labels shape: ", uci_labels.shape)

    scaler = StandardScaler()
    uci_features = scaler.fit_transform(uci_features)
    
    
    # ---------------- Training ---------------- #

    X_train, X_val, y_train, y_val = train_test_split(uci_features, uci_labels, test_size=0.2, random_state=42)

    base = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror'
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    print("Starting training...")

    Y_pred = model.predict(X_val)
    mse = np.mean((Y_pred - y_val) ** 2)
    var = np.var(y_val)

    non_zero_count = np.count_nonzero(y_val)
    mask = y_val != 0
    within_dist_count = np.sum(np.abs(Y_pred[mask] - y_val[mask]) < DIST)
    accuracy = within_dist_count / non_zero_count
    print(f"Validation Variance: {var:.4f}")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation Accuracy (within {DIST} for non-zero vals): {accuracy:.4f}")

    print("Training complete.")