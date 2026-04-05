import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error


def build_features(df: pd.DataFrame) -> tuple:
    """
    Use adstock-transformed and Hill-saturated features instead of raw spend.
    This is the critical upgrade — the model now sees diminishing returns
    baked into the input features rather than assuming linearity.
    """
    # Prefer transformed features if available
    transformed_available = [
        c for c in df.columns if c.endswith("_transformed")
    ]

    if transformed_available:
        feature_cols = transformed_available + [
            "tv_sov", "week_of_year", "month",
            "sin_week", "cos_week",
        ]
    else:
        feature_cols = [
            "tv_S", "facebook_S", "search_S", "ooh_S",
            "tv_rolling4", "tv_sov", "week_of_year", "month",
        ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["revenue"]

    return X, y, feature_cols


def train_xgb(df: pd.DataFrame) -> tuple:
    """
    Train XGBoost with time-series cross-validation.

    Critical: never use random CV splits on time series.
    Always split chronologically — future data cannot inform past predictions.
    """
    X, y, feature_cols = build_features(df)

    tscv = TimeSeriesSplit(n_splits=5)
    mapes = []

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        mapes.append(mean_absolute_percentage_error(y_val, preds))

    # Final fit on all data
    model.fit(X, y)

    # SHAP values for explainability
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return model, feature_cols, float(np.mean(mapes)), shap_values, X