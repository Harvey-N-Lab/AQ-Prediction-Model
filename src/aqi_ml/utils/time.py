from typing import Tuple
def time_split(df, time_col: str, split_ratio: float = 0.8) -> Tuple:
    # Simple chronological split (optional helper; not used in config-driven path)
    df = df.sort_values(time_col)
    n = int(len(df) * split_ratio)
    return df.iloc[:n], df.iloc[n:]
