import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from mono_dense_keras import MonoDense

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
rng = np.random.default_rng(42)


def constrained_nn(X,y, monotone_constraints):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=41
    )
    model = Sequential()
    monotone_constraints = list(monotone_constraints.values())
    model.add(Input(shape=(len(monotone_constraints),)))
    monotonicity_indicator = monotone_constraints
    model.add(
        MonoDense(128, activation="elu", monotonicity_indicator=monotonicity_indicator)
    )
    model.add(MonoDense(16, activation="elu"))
    model.add(MonoDense(16, activation="elu"))
    model.add(MonoDense(16, activation="elu"))
    model.add(MonoDense(1))

    print(model.summary())


    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10000 // 32,
        decay_rate=0.99,
    )
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="mse")

    model.fit(
        x=X_train, y=y_train, batch_size=32, validation_data=(X_test, y_test), epochs=50
    )
    df_results = pd.DataFrame({
        "y_test": y_test,
        "y_pred": np.ravel(model.predict(X_test))
    })
    print(df_results.tail(50))
    return model