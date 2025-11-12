import pandas as pd


def select_features(df: pd.DataFrame, cols: list, drop_na: bool = True) -> pd.DataFrame:
    """
    Select a defined feature set, with strict validation:
    - raises KeyError if any required feature is missing
    - raises TypeError if any selected feature is non-numeric
    - optionally drops rows with NA in the selected features
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    out = df[cols].copy()
    if drop_na:
        out = out.dropna()

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(out[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric features present: {non_numeric}")

    return out
