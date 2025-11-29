# Set the stage
import numpy as np
import torch 
import matplotlib.pyplot as plt
from SMM.SmoothMonotonicNN import SmoothMonotonicNN
from SMM.SMM_MLP import SMM_MLP
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def SMM(X,y,monotone_constraints):
    monotone_constraints = np.array(list(monotone_constraints.values()))
    X = X * monotone_constraints
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=41
    )
    # Convert to FloatTensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)

# Create model
    mask = np.array([i for i in range(len(monotone_constraints))])
    
    K = 6  # neurons per group
    model = SMM_MLP(dim=len(monotone_constraints), increasing=mask, num_neuron=32, K = K, transform="exp", last_linear=True)

    # Optimize model
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    loss_function = torch.nn.MSELoss()

    max_iterations = 200  # Numper of epochs
    for epoch in tqdm(range(max_iterations), desc="Training", ncols=100):

        # --- Training step ---
        model.train()
        optimizer.zero_grad()

        y_hat = model(X_train)
        train_loss = loss_function(y_hat, y_train)
        train_loss.backward()
        optimizer.step()

        # --- Test step ---
        model.eval()
        with torch.no_grad():
            y_test_hat = model(X_test)
            test_loss = loss_function(y_test_hat, y_test)

        tqdm.write(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss.item():.6f} | "
                f"Test Loss: {test_loss.item():.6f}")
    return model
    