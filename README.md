
# Yield Curve Forecasting (ECB Spot Rates)

This project forecasts euro-area zero-coupon (spot) yields at 2Y, 5Y, and 10Y maturities and compares:
- Naïve baseline (last observation)
- Random Forest
- XGBoost

All models are evaluated using a rolling-window, one-step-ahead forecasting framework with fixed training windows.
This setup mimics real-time forecasting conditions, ensures reproducibility, and avoids look-ahead bias.

All file paths are defined relative to the project root to ensure portability across environments.

## Setup
Create environment (conda):
```bash
conda env create -f environment.yml
conda activate yield-curve-forecasting
Python version: 3.10+


### Optional dependency
XGBoost is used for one of the machine learning models.
If it is not available, the Random Forest and baseline models will still run.

To enable XGBoost, ensure that `xgboost` is installed via `environment.yml`
or `requirements.txt`.

Note: The Diebold–Mariano test is implemented using a normal approximation
without HAC adjustment and should be interpreted as indicative rather than
definitive.


## Results
Running `python main.py` generates prediction-vs-actual plots for
2Y, 5Y, and 10Y yields using a baseline, Random Forest, and XGBoost model.
All outputs are saved in the `results/` directory.
>>>>>>> 029359e (Finalize project: reproducible pipeline, clean entry point, updated README)
