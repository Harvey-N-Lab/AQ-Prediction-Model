import argparse, json, os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

def load_dataset(path):
    return pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = load_dataset(args.train)
    val   = load_dataset(args.val)

    target = "target_pm25_next_1h"
    y_tr = train.pop(target)
    y_va = val.pop(target)

    dtr = xgb.DMatrix(train, label=y_tr)
    dva = xgb.DMatrix(val,   label=y_va)

    params = {
        "objective":"reg:squarederror",
        "eval_metric":"rmse",
        "eta":0.1,
        "max_depth":6,
        "subsample":0.8,
        "colsample_bytree":0.8
    }
    bst = xgb.train(params, dtr, num_boost_round=500, evals=[(dtr,"train"),(dva,"val")], early_stopping_rounds=30)

    pred = bst.predict(dva)
    rmse = sqrt(mean_squared_error(y_va, pred))
    mape = mean_absolute_percentage_error(y_va, pred)

    os.makedirs(args.out, exist_ok=True)
    bst.save_model(os.path.join(args.out, "model.xgb"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"rmse": rmse, "mape": mape}, f, indent=2)

if __name__ == "__main__":
    main()
