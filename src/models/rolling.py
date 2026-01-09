"""
Rolling-window one-step-ahead forecasting.
"""

from __future__ import annotations
import numpy as np


def rolling_forecast(x, y, model_factory, train_size: int = 72):
    """
    1-step ahead rolling window forecasting:
    - train on [t-train_size, ..., t-1]
    - predict at t
    """
    y_true, y_pred = [], []
    n_obs = len(x)

    for t in range(train_size, n_obs):
        x_train = x.iloc[t - train_size : t]
        y_train = y.iloc[t - train_size : t]
        x_test = x.iloc[t : t + 1]
        y_test = y.iloc[t : t + 1]

        model = model_factory()
        model.fit(x_train, y_train)
        pred = float(np.asarray(model.predict(x_test))[0])

        y_true.append(float(y_test.values[0]))
        y_pred.append(pred)

    return y_true, y_pred
