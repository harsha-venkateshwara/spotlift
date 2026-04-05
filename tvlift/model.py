import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import shap

def build_features(df: pd.DataFrame) -> tuple:
    feature_cols = [
        "tv_S", "facebook_S", "search_S", "ooh_S",
        "tv_S_lag1", "tv_S_lag2",
        "tv_rolling4", "tv_sov",
        "week_of_year", "month"
    ]
    X = df[feature_cols].fillna(0)
    y = df["revenue"]
    return X, y, feature_cols


def train_xgb(df: pd.DataFrame):
    X, y, feature_cols = build_features(df)

    # Time-series cross-validation (never shuffle time series!)
    tscv = TimeSeriesSplit(n_splits=5)
    mapes = []

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        preds = model.predict(X_val)
        mapes.append(mean_absolute_percentage_error(y_val, preds))

    # Final fit on all data
    model.fit(X, y)

    # SHAP values for explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return model, feature_cols, np.mean(mapes), shap_values, X