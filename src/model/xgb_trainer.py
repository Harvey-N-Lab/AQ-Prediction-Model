import os, json
import xgboost as xgb
import pandas as pd
from typing import Any
from aqi_ml.ports import ModelTrainerPort
from aqi_ml.utils.io import ensure_dir

class XGBTrainer(ModelTrainerPort):
    def __init__(self, params: dict, num_boost_round: int, early_stopping_rounds: int):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def train(self, X_tr, y_tr, X_val, y_val) -> Any:
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_val, label=y_val)
        boosters = xgb.train(
            self.params, dtr,
            num_boost_round=self.num_boost_round,
            evals=[(dtr,"train"),(dva,"val")],
            early_stopping_rounds=self.early_stopping_rounds
        )
        return boosters

    def predict(self, model, X):
        d = xgb.DMatrix(X)
        return model.predict(d)

    def save(self, model, out_dir: str) -> None:
        ensure_dir(out_dir)
        model.save_model(os.path.join(out_dir, "model.xgb"))
