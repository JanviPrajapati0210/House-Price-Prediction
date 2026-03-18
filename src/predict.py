import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(__file__))

from preprocess import preprocess

def predict():
    print("loading model & data...")

    # load saved model
    model = joblib.load('../models/model.pkl')

    # preprocess to get test data
    _, X_test, _ = preprocess('../data/train.csv', '../data/test.csv')

    # predict - model was trained on log prices so convert back
    log_preds  = model.predict(X_test)
    predictions = np.expm1(log_preds)  # reverse the log transform

    # save to outputs
    os.makedirs('../outputs', exist_ok=True)
    output = pd.DataFrame({
        'Id'       : range(1461, 1461 + len(predictions)),
        'SalePrice': predictions.astype(int)
    })
    output.to_csv('../outputs/predictions.csv', index=False)

    print(f"predictions saved to outputs/predictions.csv")
    print(f"\nsample predictions:")
    print(output.head(10).to_string(index=False))

    # basic sanity check
    print(f"\nprice range : ${predictions.min():,.0f} - ${predictions.max():,.0f}")
    print(f"average price : ${predictions.mean():,.0f}")

if __name__ == "__main__":
    predict()