"""
SageMaker-compatible inference module.

Expected layout inside model.tar.gz:
- model.xgb
- features.json or schema.json (list of allowed features)
- code/inference.py  (this file)
"""

import json
import os
from io import StringIO
from typing import List, Optional

import pandas as pd
import xgboost as xgb

_model: Optional[xgb.Booster] = None
_ALLOWED: Optional[List[str]] = None


def model_fn(model_dir: str):
    """
    Load the trained Booster and (optionally) the allowed input schema.
    """
    global _model, _ALLOWED
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "model.xgb"))
    _model = booster

    schema_file = None
    for candidate in ("schema.json", "features.json"):
        p = os.path.join(model_dir, candidate)
        if os.path.exists(p):
            schema_file = p
            break

    if schema_file:
        try:
            with open(schema_file, "r") as f:
                data = json.load(f)
            _ALLOWED = data.get("features") if isinstance(data, dict) else data
        except Exception:
            _ALLOWED = None

    return _model


def input_fn(request_body: str, request_content_type: str = "application/json"):
    """
    Parse the request to a DataFrame.
    Enforce the schema (if present) and preserve column order.
    """
    if request_content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), header=None)
    else:
        obj = json.loads(request_body)
        df = pd.DataFrame([obj]) if isinstance(obj, dict) else pd.DataFrame(obj)

    if _ALLOWED:
        missing = [c for c in _ALLOWED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        df = df[_ALLOWED]

    return df


def predict_fn(input_data: pd.DataFrame, model: xgb.Booster):
    dmat = xgb.DMatrix(input_data)
    return model.predict(dmat)


def output_fn(prediction, content_type: str = "application/json"):
    return json.dumps({"predictions": prediction.tolist()})
