import argparse
import torch
import torch.nn.functional as F
import numpy as np
from modules import TCBL
import time
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler
from xgboost import plot_tree
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import prod

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

    # generated=True ==> use generated data
    uci_ds = UCIDataset(generated=True)
    # uci_ds.numerize() # only use numerize if you're not augmenting -- converts data to numeric
    uci_ds.augment() # implicitly also converts data to numeric, which is why numerize() should not be used here
    concept_ds = ConceptDataset(generated=True)

    if len(uci_ds) != len(concept_ds):
        print(f"Dataset mismatch: UCIDataset={len(uci_ds)} vs ConceptDataset={len(concept_ds)}")
        exit(0)

    print("\nDatasets loaded successfully. Datasets have same number of rows.")

    # Convert entire dataset into tensors so normalization only happens **once**
    uci_features = np.vstack([uci_ds[i][0] for i in range(len(uci_ds))])
    uci_features = np.nan_to_num(uci_features, nan=0.0, posinf=1e6, neginf=-1e6)

    uci_labels = np.vstack([concept_ds[i][0] for i in range(len(concept_ds))])

    print("uci features shape: ", uci_features.shape, " uci labels shape: ", uci_labels.shape)

    print("column headers before: \n", uci_labels[1])

    scaler = StandardScaler()
    uci_features = scaler.fit_transform(uci_features)
    
    
    # ---------------- Training ---------------- #

    X_train, X_val, y_train, y_val = train_test_split(uci_features, uci_labels, test_size=0.2, random_state=42)

    train_label = y_train[:, -1] # for last column
    y_train = y_train[:, :-1] # for all but last column
    
    test_label = y_val[:, -1] # for last column
    y_val = y_val[:, :-1] # for all but last column

    print("column headers after: \n", y_train[1])

    base = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror'
    )
    print("Starting training...")
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    
    
    # # Regular
    # np.savetxt('concept_bottleneck/data_split/train_features.csv', X_train, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/train_concepts.csv', model.predict(X_train), delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/test_features.csv', X_val, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/test_concepts.csv', y_pred, delimiter=',')

    # np.savetxt('concept_bottleneck/data_split/train_labels.csv', train_label, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/test_labels.csv', test_label, delimiter=',')

    # # Augmented
    # np.savetxt('concept_bottleneck/data_split/augmented_train_features.csv', X_train, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/augmented_train_concepts.csv', model.predict(X_train), delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/augmented_test_features.csv', X_val, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/augmented_test_concepts.csv', y_pred, delimiter=',')

    # np.savetxt('concept_bottleneck/data_split/augmented_train_labels.csv', train_label, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/augmented_test_labels.csv', test_label, delimiter=',')

    # # Generated (random between training and testing)
    # np.savetxt('concept_bottleneck/data_split/generated_train_features.csv', X_train, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/generated_train_concepts.csv', model.predict(X_train), delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/generated_test_features.csv', X_val, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/generated_test_concepts.csv', y_pred, delimiter=',')

    # np.savetxt('concept_bottleneck/data_split/generated_train_labels.csv', train_label, delimiter=',')
    # np.savetxt('concept_bottleneck/data_split/generated_test_labels.csv', test_label, delimiter=',')

    # Augmented + Generated
    np.savetxt('concept_bottleneck/data_split/both_train_features.csv', X_train, delimiter=',')
    np.savetxt('concept_bottleneck/data_split/both_train_concepts.csv', model.predict(X_train), delimiter=',')
    np.savetxt('concept_bottleneck/data_split/both_test_features.csv', X_val, delimiter=',')
    np.savetxt('concept_bottleneck/data_split/both_test_concepts.csv', y_pred, delimiter=',')

    np.savetxt('concept_bottleneck/data_split/both_train_labels.csv', train_label, delimiter=',')
    np.savetxt('concept_bottleneck/data_split/both_test_labels.csv', test_label, delimiter=',')




    mse = np.mean((y_pred - y_val) ** 2)
    var = np.var(y_val)
    general_accuracy = np.sum(np.abs(y_pred - y_val) < DIST) / prod(y_val.shape)

    non_zero_count = np.count_nonzero(y_val)
    mask = y_val != 0
    within_dist_count = np.sum(np.abs(y_pred[mask] - y_val[mask]) < DIST)
    accuracy = within_dist_count / non_zero_count
    print(f"Validation Variance: {var:.4f}")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation Accuracy (within {DIST} for non-zero vals): {accuracy:.4f}")
    print(f"Validation Accuracy (within {DIST} for all vals): {general_accuracy:.4f}")

    # output_dir = "tree_plots"
    # os.makedirs(output_dir, exist_ok=True)

    # for i, est in enumerate(model.estimators_):
    #     booster = est.get_booster()

    #     plt.figure(figsize=(30, 20))
    #     xgb.plot_tree(booster, tree_idx=0)   # matplotlib-only mode
    #     plt.title(f"Tree for output concept {i}")

    #     out_path = os.path.join(output_dir, f"concept_{i}_tree.png")
    #     plt.savefig(out_path, dpi=300, bbox_inches="tight")
    #     plt.close()
    print("Training complete.")