import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def naive_attribution(df: pd.DataFrame) -> pd.DataFrame:
    spend_cols = ["tv_S","facebook_S","search_S","ooh_S","print_S"]
    total_spend = df[spend_cols].sum(axis=1)
    attribution = {}
    for col in spend_cols:
        share = df[col] / (total_spend + 1e-9)
        attribution[col] = (share * df["revenue"]).sum()
    total = sum(attribution.values())
    return {k: v/total for k, v in attribution.items()}

def ols_attribution(df: pd.DataFrame):
    """OLS regression- revenue ~ spend channels + seasonality. Shows correlation but not causiation"""

    features = ["tv_S", "facebook_S", "search_S","ooh_S","tv_rolling4","tv_sov","week_of_year","month"]

    x = df[features].fillna(0)
    y = df["revenue"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression().fit(X_Scaled, y)
    preds = model.predict(X_scaled)
    r2 = r2_score(y, preds)

    coefs = dict(zip(features, model.coef_))
    return model, coefs, r2, scaler

def compute_incremental_roas(df: pd.DataFrame, coefs: dict) -> dict:
    """Incremental ROAS = revenue attributable to spend / spend.
    Uses OLS coefficients leading to first-pass estimate """

    results={}
    for channel in ["tv_S","facebook_S", "search_S"]:
        if channel in coefs:
            incremental_rev = coefs[channel] * df[channel].std()
            roas = incremental_rev / (df[channel].mean() + 1e-9)
            results[channel] = round(roas, 3)
    return results