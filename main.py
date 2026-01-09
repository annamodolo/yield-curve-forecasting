"""
Entry point for the yield curve forecasting project.

Run:
    python main.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.data_loader import load_all_yields, to_monthly_end_of_month
from src.features import add_features, make_supervised
from src.models.baseline import naive_last_observation
from src.models.ml_models import get_random_forest, get_xgboost
from src.models.rolling import rolling_forecast
from src.models.evaluation import regression_metrics, diebold_mariano
from src.visualization import plot_pred_vs_actual

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_for_target(df_monthly: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Run rolling one-step-ahead forecasts for one target yield and return metrics."""
    df_feat = add_features(df_monthly, n_lags=1)
    X, y = make_supervised(  # pylint: disable=invalid-name
        df_feat,
        target_col=target_col,
        horizon=1,
    )

    # Prevent trivial leakage: don’t include the current target value as a feature
    X = X.drop(columns=[target_col]) # pylint: disable=invalid-name

    # Dataset size diagnostics (added)
    print(
        f"{target_col}: "
        f"monthly obs={len(df_monthly)}, "
        f"after features X={X.shape}, y={y.shape}"
    )

    train_size = 72  # 6 years of monthly data

    # --- Baseline ---
    y_true_base, y_pred_base = [], []
    for t in range(train_size, len(X)):
        y_train = y.iloc[t - train_size : t]
        y_test = y.iloc[t : t + 1]
        pred = naive_last_observation(y_train)

        y_true_base.append(float(y_test.values[0]))
        y_pred_base.append(float(pred))

    base_metrics = regression_metrics(y_true_base, y_pred_base)

    # --- Random Forest ---
    y_true_rf, y_pred_rf = rolling_forecast(
        X,
        y,
        model_factory=lambda: get_random_forest(random_state=42),
        train_size=train_size,
    )
    rf_metrics = regression_metrics(y_true_rf, y_pred_rf)

    # --- XGBoost ---
    y_true_xgb, y_pred_xgb = rolling_forecast(
        X,
        y,
        model_factory=lambda: get_xgboost(random_state=42),
        train_size=train_size,
    )
    xgb_metrics = regression_metrics(y_true_xgb, y_pred_xgb)

    # DM tests vs baseline
    e_base = (pd.Series(y_true_base) - pd.Series(y_pred_base)).values
    e_rf = (pd.Series(y_true_rf) - pd.Series(y_pred_rf)).values
    e_xgb = (pd.Series(y_true_xgb) - pd.Series(y_pred_xgb)).values

    dm_rf = diebold_mariano(e_rf, e_base)
    dm_xgb = diebold_mariano(e_xgb, e_base)

    # Save plots
    plot_pred_vs_actual(
        y_true_base,
        y_pred_base,
        title=f"{target_col} - Baseline predicted vs actual",
        out_path=str(RESULTS_DIR / f"{target_col}_baseline_pred_vs_actual.png"),
    )
    plot_pred_vs_actual(
        y_true_rf,
        y_pred_rf,
        title=f"{target_col} - RandomForest predicted vs actual",
        out_path=str(RESULTS_DIR / f"{target_col}_rf_pred_vs_actual.png"),
    )
    plot_pred_vs_actual(
        y_true_xgb,
        y_pred_xgb,
        title=f"{target_col} - XGBoost predicted vs actual",
        out_path=str(RESULTS_DIR / f"{target_col}_xgb_pred_vs_actual.png"),
    )

    rows = [
        {"target": target_col,
        "model": "baseline_naive", **base_metrics, "DM": None, "p_value": None},
        {"target": target_col, "model": "random_forest", **rf_metrics, **dm_rf},
        {"target": target_col, "model": "xgboost", **xgb_metrics, **dm_xgb},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Load data, run forecasts for each maturity, and save metrics/plots."""
    df_daily = load_all_yields()
    df_monthly = to_monthly_end_of_month(df_daily)

    all_results = []
    for target in ["yield_2y", "yield_5y", "yield_10y"]:
        all_results.append(run_for_target(df_monthly, target))

    results_df = pd.concat(all_results, ignore_index=True)
    print("\n=== METRICS SUMMARY ===")
    print(results_df)

    out_csv = RESULTS_DIR / "metrics_summary.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to: {out_csv}")


if __name__ == "__main__":
    main()
