"""
Baseline model: last observation carried forward.
"""

from __future__ import annotations
import numpy as np


def naive_last_observation(y_train) -> float:
    """Predict next value as last observed value."""
    return float(np.asarray(y_train)[-1])
