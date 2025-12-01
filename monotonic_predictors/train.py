import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from simple_models import XGB, sweep_feature, LGB, CAT
from constrained_nn import constrained_nn 
from smm_nn import SMM
train_f = pd.read_csv("/home/tsaxena/10747/MonoCo/concept_bottleneck/data_split/train_concepts.csv")
train_l = pd.read_csv("/home/tsaxena/10747/MonoCo/concept_bottleneck/data_split/train_labels.csv")
test_f = pd.read_csv("/home/tsaxena/10747/MonoCo/concept_bottleneck/data_split/test_concepts.csv")
test_l = pd.read_csv("/home/tsaxena/10747/MonoCo/concept_bottleneck/data_split/test_labels.csv")

X_train = train_f.to_numpy(dtype=float)
y_train = train_l.to_numpy(dtype=float)
y_train = y_train / 4.0

X_test = test_f.to_numpy(dtype=float)
y_test = test_l.to_numpy(dtype=float)
y_test = y_test / 4.0

monotone_constraints = {}
for i in range(20):
    if i < 10:
        monotone_constraints["x" + str(i)] = +1
    else:
        monotone_constraints["x" + str(i)] = -1

model_lgb = SMM(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,monotone_constraints=monotone_constraints)
i = 1 # pick a sample
x = X_train[i].copy()

xs, preds = sweep_feature(model_lgb, x, f_index=5)

import matplotlib.pyplot as plt
plt.plot(xs, preds)
plt.title("Feature 0 sweep (should be monotone increasing)")
plt.show()


