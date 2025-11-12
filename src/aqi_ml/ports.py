from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import pandas as pd

class DataLoaderPort(ABC):
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame: ...

class ModelTrainerPort(ABC):
    @abstractmethod
    def train(self, X_tr, y_tr, X_val, y_val) -> Any: ...
    @abstractmethod
    def predict(self, model, X) -> Any: ...
    @abstractmethod
    def save(self, model, out_dir: str) -> None: ...

class RegistryPort(ABC):
    @abstractmethod
    def register(self, model_artifact_dir: str, metrics: dict) -> None: ...
