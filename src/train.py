import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from preprocess import preprocess
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

def evaluate_model(model, X_val, y_val, model_name):
    preds = model.predict(X_val)
    
    # convert back from log scale to actual prices for readable metrics
    preds_actual = np.expm1(preds)
    y_actual     = np.expm1(y_val)
    
    mae  = mean_absolute_error(y_actual, preds_actual)
    rmse = np.sqrt(mean_squared_error(y_actual, preds_actual))
    r2   = r2_score(y_val, preds)  # r2 stays on log scale, more stable
    
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")
    return rmse

def train():
    print("loading & preprocessing data...")
    X_train, X_test, y = preprocess('../data/train.csv', '../data/test.csv')

    # log transform target - helps with skewness we saw in eda
    y_log = np.log1p(y)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_log, test_size=0.2, random_state=42
    )
    print(f"  train size : {X_tr.shape}")
    print(f"  val size   : {X_val.shape}")

    results = {}

    # model 1 - linear regression, simple baseline
    print("\nlinear regression:")
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    results['LinearRegression'] = evaluate_model(lr, X_val, y_val, 'lr')

    # model 2 - random forest
    print("\nrandom forest:")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    results['RandomForest'] = evaluate_model(rf, X_val, y_val, 'rf')

    # model 3 - xgboost, usually the best for this kind of data
    print("\nxgboost:")
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0  # dont print xgb logs
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    results['XGBoost'] = evaluate_model(xgb, X_val, y_val, 'xgb')

    # pick best model based on rmse
    best_model_name = min(results, key=results.get)
    print(f"\nbest model : {best_model_name} (lowest RMSE)")

    # save best model
    best_model = {'LinearRegression': lr, 'RandomForest': rf, 'XGBoost': xgb}[best_model_name]
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/model.pkl')

   
    with open('../models/best_model_name.txt', 'w') as f:
        f.write(best_model_name)

    print(f"model saved to models/model.pkl")

if __name__ == "__main__":
    train()