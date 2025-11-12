import json
import os
import platform
import random
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from aqi_ml.config import AppConfig
from aqi_ml.data.features import select_features
from aqi_ml.data.loader import PandasDataLoader
from aqi_ml.model.registry import NoopRegistry
from aqi_ml.model.xgb_trainer import XGBTrainer
from aqi_ml.utils.io import ensure_dir
from aqi_ml.utils.logging import get_logger


@dataclass
class TrainResult:
    artifacts_dir: str
    rmse: float
    mape: float


class TrainingPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.logger = get_logger("training")

        # Ports/adapters
        self.loader = PandasDataLoader()
        # Extend later with a real registry; Noop keeps the port in place
        self.registry = NoopRegistry() if not cfg.registry.enabled else NoopRegistry()

    def _seed(self, seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)

    def run(self) -> TrainResult:
        c = self.cfg
        self.logger.info(json.dumps({"event": "experiment_start", "name": c.experiment.name}))

        self._seed(c.experiment.seed)

        # Load
        df_tr = self.loader.load(c.data.train_path)
        df_va = self.loader.load(c.data.val_path)

        # Split X/y
        target = c.experiment.target
        feat_cols = c.data.features.include

        X_tr = select_features(df_tr, feat_cols, drop_na=c.data.features.drop_na)
        y_tr = df_tr.loc[X_tr.index, target]

        X_va = select_features(df_va, feat_cols, drop_na=c.data.features.drop_na)
        y_va = df_va.loc[X_va.index, target]

        # Trainer
        if c.trainer.kind.lower() == "xgboost":
            trainer = XGBTrainer(
                c.trainer.params, c.trainer.num_boost_round, c.trainer.early_stopping_rounds
            )
        else:
            raise ValueError(f"Unknown trainer kind: {c.trainer.kind}")

        # Train
        t0 = time.time()
        model = trainer.train(X_tr, y_tr, X_va, y_va)

        # Evaluate
        yhat = trainer.predict(model, X_va)
        rmse = sqrt(mean_squared_error(y_va, yhat))
        mape = mean_absolute_percentage_error(y_va, yhat)
        metrics = {"rmse": rmse, "mape": mape}
        self.logger.info(json.dumps({"event": "validation_metrics", **metrics}))

        # Save artifacts
        out_dir = Path(c.artifacts.output_dir)
        ensure_dir(out_dir)
        trainer.save(model, str(out_dir))

        if c.artifacts.save_feature_list:
            (out_dir / "features.json").write_text(json.dumps({"features": feat_cols}, indent=2))

        # Save predictions for quick inspection
        pred_path = out_dir / "val_predictions.csv"
        pd.DataFrame({"y_true": y_va.values, "y_pred": yhat}).to_csv(pred_path, index=False)

        # Save metrics.json (SageMaker/CI-friendly)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Run manifest for reproducibility
        run_meta = {
            "experiment": c.experiment.name,
            "seed": c.experiment.seed,
            "trainer": c.trainer.kind,
            "params": c.trainer.params,
            "python": platform.python_version(),
            "time_start": t0,
            "time_end": time.time(),
            "metrics": metrics,
        }
        (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2))

        # Optional: register to a registry
        self.registry.register(str(out_dir), metrics)

        self.logger.info(json.dumps({"event": "artifacts_saved", "dir": str(out_dir)}))
        return TrainResult(str(out_dir), rmse, mape)
