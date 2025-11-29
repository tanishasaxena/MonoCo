import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from simple_models import XGB, sweep_feature, LGB, CAT
from constrained_nn import constrained_nn 
from smm_nn import SMM
heart_disease_data = pd.read_csv("/home/iron/code/MonoCo/data/heart_disease/heart_disease_uci_concepts.csv")
feature_cols = heart_disease_data.columns[:-1]      # all except last, adjust if needed
label_col = "label"

X = heart_disease_data[feature_cols].to_numpy(dtype=float)
y = heart_disease_data["label"].to_numpy(dtype=float)
y = y / 4.0

monotone_constraints = {}
for i in range(20):
    if i < 10:
        monotone_constraints["x" + str(i)] = +1
    else:
        monotone_constraints["x" + str(i)] = -1

model_lgb = SMM(X = X,y = y,monotone_constraints=monotone_constraints)
i = 1 # pick a sample
x = X[i].copy()

xs, preds = sweep_feature(model_lgb, x, f_index=5)

import matplotlib.pyplot as plt
plt.plot(xs, preds)
plt.title("Feature 0 sweep (should be monotone increasing)")
plt.show()


