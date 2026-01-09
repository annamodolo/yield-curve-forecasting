PROJECT PROPOSAL - Anna Modolo

Project title and cathegory
• Title: “Yield Curve Forecasting with Machine Learning”
• Cathegory: Data analysis, Finance, Statistical Modeling, Machine Learning

Problem statement or Motivation
In fixed-income market, the yield curve, that shows the relationship between bond yields and
maturities, reflects investors expectations for growth, inflation, and monetary policy. It is important
to forecast its movements in order to value default free bonds and manage interest rate risk.
This project aims to predict future movements of the risk free yield curve using Random Forest and
XGBoost, comparing their forecasting performance to indetify which
technique best models yield changes across different maturities and market conditions.
Planned approach and technologies

The project will be implemented in Python and I will be using:
• pandas / NumPy for data handling
• scikit-learn for Random Forest and XGBoost models
• TensorFlow (Keras) for LSTM recurrent networks
• matplotlib / seaborn for visualization
• pytest for testing reproducibility


The main steps are going to be:
1. Data collection and preprocessing: download zero-coupon yield curve data and €STR short rate
data from the ECB Statistical Data Warehouse (2004-present) and clean. Use monthly data as
this frequency provides a smoother and more stable representation of interest rate dynamics.
Then interpolate missing maturities and create features such as yield-curve slopes, curvatures
and principal components (PCA).

2. Model development: Train Random Forest, XGBoost, and LSTM models to predict next-period
yield for each maturity.

3. Evaluation and Compariosn: Use rolling-window forecasts to measure accuracy with metrics
such as RMSE, MAE, and R^2. Also apply Diebold-Mariano tests to check wheter differences
in forecast errors are statistically significant. To properly evaluate forecast accuracy, I will
include a simple baseline benchmark: a naïve “last observation carried forward” predictor (i.e.,
assuming next month’s yield equals the current month’s yield). All machine-learning models
(Random Forest, XGBoost) will be compared against this baseline using the same
rolling-window forecast framework.

4. Visualization: Plot predicted vs actual yield curves, error heatmaps, and feature-importance
rankings.

5. Analysis: Examine which maturities and market regimes each model handles best.
Expected challenges and how I will adress them.

1. Data volume and quality: handle missing or irregular maturities using interpolation and scaling.

2. Overfitting risk: use cross-validation, regularization, and irregular maturities using interpolation
and scaling.

3. Comparability: apply the same rolling-wondow framework and metrics across all model to
ensure fairness.


Success criteria
The project will be successful if:
• Real ECB yield curve data are cleaned and structured for machine learning use.
• Random Forest and XGBoost models are trained and evaluated on multiple maturities.
• Machine-learning models significantly outperform the naïve benchmark according to RMSE,
MAE, and Diebold–Mariano tests.
• Visualizations effectively demonstrate model accuracy and limitations.

Strech goals (if time permits)
• Add macro-economi variables such as inflation and GDP growth.
• Extend to U.S. Treasury data for cross market validation.
• Deploy an interactive dashboard (Plotly Dash) for real-time yield curve forecast.
