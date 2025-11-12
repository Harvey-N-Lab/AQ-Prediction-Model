from aqi_ml.data.features import select_features
import pandas as pd
import pytest


def test_select_features_valid():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = select_features(df, ["a", "b"])
    assert list(out.columns) == ["a", "b"]


def test_select_features_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        select_features(df, ["a", "b"])


def test_select_features_non_numeric():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    with pytest.raises(TypeError):
        select_features(df, ["a", "b"])
