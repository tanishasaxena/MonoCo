import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, DMatrix
import lightgbm as lgb
from catboost import CatBoostRegressor

def XGB(X_train,y_train,X_test,y_test,monotone_constraints):
    # feature_names = [f"x{i}" for i in range(X.shape[1])]

    # dfX = pd.DataFrame(X, columns=feature_names)

    # X_train, X_test, y_train, y_test = train_test_split(dfX, y, test_size=0.1, random_state=1)
    
    # Convert dict monotone_constraints to tuple ordered by feature index
    if isinstance(monotone_constraints, dict):
        n_features = X_train.shape[1]
        monotone_constraints = tuple(
            monotone_constraints.get(f"x{i}", 0) for i in range(n_features)
        )
    
    model = XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        monotone_constraints=monotone_constraints,
        tree_method="hist",
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("XGBoost MSE on train:", mean_squared_error(y_train, model.predict(X_train)))
    print("XGBoost MSE on test:", mean_squared_error(y_test, pred))
    df_results = pd.DataFrame({
        "y_test": y_test.flatten(),
        "y_pred": pred
    })
    print(df_results.head(20))
    return model

def sweep_feature(model, x, f_index, low=0, high=1, steps=200):
    xs = []
    preds = []

    grid = np.linspace(low, high, steps)
    for v in grid:
        x_mod = x.copy()
        x_mod[f_index] = v
        xs.append(v)
        preds.append(model.predict(x_mod.reshape(1, -1))[0])

    return np.array(xs), np.array(preds)


def LGB(X_train,y_train,X_test,y_test,monotone_constraints):
    # feature_names = [f"x{i}" for i in range(X.shape[1])]
    # dfX = pd.DataFrame(X, columns=feature_names)
    monotone_constraints = list(monotone_constraints.values())

    # X_train, X_test, y_train, y_test = train_test_split(
    #     dfX, y, test_size=0.1, random_state=41
    # )

    train_data = lgb.Dataset(X_train, label=y_train)

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.02,
        "num_leaves": 12,
        "monotone_constraints": monotone_constraints,
        "verbosity": -1,
        "force_row_wise": True,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
    )

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    print("LightGBM MSE on train:", mean_squared_error(y_train, pred_train))
    print("LightGBM MSE on test:",  mean_squared_error(y_test, pred_test))
    # df_results = pd.DataFrame({
    #     "y_test": y_test,
    #     "y_pred": pred_test
    # })
    # print(df_results.tail(50))
    return model


def CAT(X_train,y_train,X_test,y_test,monotone_constraints):
    # Same as XGB/LGB wrapper
    # feature_names = [f"x{i}" for i in range(X.shape[1])]
    # dfX = pd.DataFrame(X, columns=feature_names)

    # Convert dict → monotone list if needed
    if isinstance(monotone_constraints, dict):
        monotone_constraints = list(monotone_constraints.values())

    # X_train, X_test, y_train, y_test = train_test_split(
    #     dfX, y, test_size=0.1, random_state=41
    # )

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function="RMSE",
        monotone_constraints=monotone_constraints,
        verbose=False
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    print("CatBoost MSE on train:", mean_squared_error(y_train, pred_train))
    print("CatBoost MSE on test:",  mean_squared_error(y_test, pred_test))

    return model