"""
Model factories for machine learning regressors.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor as _XGBRegressor
except ImportError:
    _XGBRegressor = None


def get_random_forest(random_state: int = 42) -> RandomForestRegressor:
    """
    Create a Random Forest regressor with fixed hyperparameters.
    """
    return RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )


def get_xgboost(random_state: int = 42):
    """
    Create an XGBoost regressor with fixed hyperparameters.
    """
    if _XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed. Add it to environment.yml or requirements.txt."
        )

    return _XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
