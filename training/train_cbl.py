import argparse
import torch
import torch.nn.functional as F
import numpy as np
from modules import TCBL
import time
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler

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
    def train():
        best_val_loss = float("inf")
        results_buffer = ""

        # for idx, path in enumerate(tqdm(paths, desc="Autoencoding + compressing Kodak", ncols=100)):
        for epoch in tqdm(range(args.epochs), desc="Training...", ncols=100):
            model.train()
            total_loss = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred_vecs = model(batch_x)
                # print(f"Pred vecs shape {pred_vecs.shape} and batch shape {batch_y.shape}")
                loss = criterion(pred_vecs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_loss, correct, count = 0, 0, 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    pred_vecs = model(batch_x)
                    loss = criterion(pred_vecs, batch_y)
                    val_loss += loss.item()
                    
                    # print(f"Pred vecs shape {pred_vecs.shape} and batch shape {batch_y.shape}")
                    # print(f"Pred vecs: {pred_vecs} \n Batch y: {batch_y}")

                    correct += (abs(pred_vecs - batch_y) < DIST and batch_y != 0).sum().item()
                    count += (batch_y != 0).sum().item()

            avg_train = total_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            accuracy = correct / count

            results_buffer += f"\nEpoch {epoch+1}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Accuracy: {accuracy:.4f}"

            # Save best model
            if avg_val < best_val_loss:
                torch.save(model.state_dict(), f"{CBL_PATH}tcbl_epoch_{epoch+1}.pt")
                best_val_loss = avg_val
                # print(f"Epoch {epoch+1}: New best model saved to {CBL_PATH}tcbl_epoch_{epoch+1}.pt")
        
        return results_buffer

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

    X = torch.tensor(uci_features, dtype=torch.float32)
    y = torch.tensor(uci_labels, dtype=torch.float32)
    
    
    # ---------------- Training ---------------- #

    dataset = torch.utils.data.TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TCBL(in_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    print("Starting training...")

    results_buffer = train()
    print(results_buffer)

    print("Training complete.")