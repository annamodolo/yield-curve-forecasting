"""
Feature engineering for yield curve forecasting (2Y, 5Y, 10Y).
"""

from __future__ import annotations
import pandas as pd


def add_features(df: pd.DataFrame, n_lags: int = 1) -> pd.DataFrame:
    """
    Add term-structure features and lagged features.
    slope = 10Y - 2Y
    curvature = 2*5Y - 2Y - 10Y
    """
    out = df.copy()

    out["slope_10y_2y"] = out["yield_10y"] - out["yield_2y"]
    out["curvature"] = 2.0 * out["yield_5y"] - out["yield_2y"] - out["yield_10y"]

    for lag in range(1, n_lags + 1):
        for col in ["yield_2y", "yield_5y", "yield_10y", "slope_10y_2y", "curvature"]:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    return out.dropna()


def make_supervised(
    df: pd.DataFrame, target_col: str, horizon: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create X, y for forecasting y_{t+horizon} from info at time t.
    """
    x = df.copy()
    y = df[target_col].shift(-horizon)

    x = x.iloc[:-horizon]
    y = y.iloc[:-horizon]

    return x, y
