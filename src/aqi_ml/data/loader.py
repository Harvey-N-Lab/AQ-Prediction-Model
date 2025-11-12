import pandas as pd
from aqi_ml.ports import DataLoaderPort
from aqi_ml.utils.io import read_table

class PandasDataLoader(DataLoaderPort):
    def load(self, path: str) -> pd.DataFrame:
        return read_table(path)
