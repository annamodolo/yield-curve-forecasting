"""
Evaluation utilities for forecasting models.
"""

from __future__ import annotations
from math import erf, sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """
    Compute standard regression metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def diebold_mariano(e_model, e_benchmark) -> dict:
    """
    Diebold–Mariano test comparing squared forecast errors.

    Uses a normal approximation for the p-value.
    """
    e_model = np.asarray(e_model, dtype=float)
    e_benchmark = np.asarray(e_benchmark, dtype=float)

    d = (e_model ** 2) - (e_benchmark ** 2)
    d = d[~np.isnan(d)]

    if len(d) < 10:
        return {"DM": np.nan, "p_value": np.nan}

    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0))))

    return {"DM": float(dm_stat), "p_value": float(p_value)}
