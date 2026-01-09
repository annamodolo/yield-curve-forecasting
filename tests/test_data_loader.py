"""
Tests for the data_loader module.
Ensures the yield curve file exists and loads correctly.
"""

import os  # standard library first
from src.data_loader import load_yield_curve  # then project imports


def test_yield_curve_file_exists():
    """Check that the raw yield curve CSV file is present."""
    assert os.path.exists("data/raw/yield_curve.csv")


def test_load_yield_curve_returns_dataframe():
    """Check that load_yield_curve loads a non-empty DataFrame."""
    df = load_yield_curve()
    assert df is not None
    assert not df.empty
