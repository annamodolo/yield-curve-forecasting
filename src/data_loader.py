"""
Data loading utilities for the ECB yield curve forecasting project.

This module reads ECB Data Portal CSV exports (semicolon-separated) for
spot/zero-coupon yields at multiple maturities and provides helpers to
convert daily data to monthly end-of-month frequency.
"""

from __future__ import annotations

import pandas as pd


def load_ecb_spot_rate(path: str, col_name: str) -> pd.DataFrame:
    """
    Load a single ECB spot-rate CSV exported from the ECB Data Portal.

    Keeps:
      - DATE
      - the value column (usually OBS_VALUE, sometimes the long series name)

    Returns a DataFrame indexed by date with one column (col_name).
    """
    df = pd.read_csv(
        path,
        sep=";",
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
    )

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # DATE column is always there
    if "DATE" not in df.columns:
        raise ValueError(f"'DATE' column not found in {path}. Columns: {df.columns.tolist()}")

    # Prefer OBS_VALUE if present; otherwise take the last column
    value_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else df.columns[-1]

    df = df[["DATE", value_col]].copy()
    df.columns = ["date", col_name]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[col_name] = (
        df[col_name]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    df = df.dropna(subset=["date", col_name]).set_index("date").sort_index()
    return df


def load_all_yields(
    path_2y: str = "data/raw/ecb_spot_2y.csv",
    path_5y: str = "data/raw/ecb_spot_5y.csv",
    path_10y: str = "data/raw/ecb_spot_10y.csv",
) -> pd.DataFrame:
    """
    Load and merge 2Y, 5Y, 10Y spot yields into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Index: date
        Columns: yield_2y, yield_5y, yield_10y
    """
    y2 = load_ecb_spot_rate(path_2y, "yield_2y")
    y5 = load_ecb_spot_rate(path_5y, "yield_5y")
    y10 = load_ecb_spot_rate(path_10y, "yield_10y")

    # Inner join keeps dates present in all three series (clean for modeling)
    df = y2.join([y5, y10], how="inner")

    return df


def to_monthly_end_of_month(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily yields to monthly frequency using end-of-month observation.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily time series with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Monthly (end-of-month) time series.
    """
    return df_daily.resample("M").last().dropna()
