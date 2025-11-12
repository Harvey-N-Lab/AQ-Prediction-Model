from pathlib import Path
import pandas as pd

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if str(p).startswith("s3://"):
        # s3fs handled by pandas if installed
        if str(p).endswith(".parquet"):
            return pd.read_parquet(path, engine="pyarrow")
        return pd.read_csv(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p, engine="pyarrow")
    return pd.read_csv(p)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)
