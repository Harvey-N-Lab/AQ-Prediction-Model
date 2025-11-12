from pydantic import BaseModel
from typing import List, Optional
import yaml, os

class Experiment(BaseModel):
    name: str
    seed: int = 42
    target: str = "target_pm25_next_1h"

class DataSection(BaseModel):
    train_path: str
    val_path: str
    class Features(BaseModel):
        include: List[str]
        drop_na: bool = True
    features: Features

class TrainerSection(BaseModel):
    kind: str = "xgboost"
    params: dict = {}
    num_boost_round: int = 500
    early_stopping_rounds: int = 30

class Artifacts(BaseModel):
    output_dir: str = "out/model"
    save_feature_list: bool = True

class Registry(BaseModel):
    enabled: bool = False

class AppConfig(BaseModel):
    experiment: Experiment
    data: DataSection
    trainer: TrainerSection
    artifacts: Artifacts
    registry: Registry = Registry()

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
