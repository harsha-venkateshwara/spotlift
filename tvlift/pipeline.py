import pandas as pd
import numpy as np
from tvlift.adstock import (
    fit_adstock_params,
    apply_adstock_and_saturation,
    geometric_adstock,
)


def load_and_engineer(path: str) -> tuple:
    """
    Load Robyn MMM dataset and apply full feature engineering:
      1. Basic lag and rolling features
      2. Fitted adstock transforms (geometric decay)
      3. Hill saturation transforms (diminishing returns)
      4. Share of voice
      5. Seasonality encoding

    Returns (df, adstock_params) so the dashboard can display
    the fitted decay rates.
    """
    df = pd.read_csv(path, parse_dates=["DATE"])
    df = df.rename(columns={"DATE": "date", "revenue": "revenue"})
    df = df.sort_values("date").reset_index(drop=True)

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["sin_week"]     = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"]     = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # ── Fit adstock parameters from data ─────────────────────────────────
    adstock_params = {}
    channels = {
        "tv_S":       "tv",
        "facebook_S": "facebook",
        "search_S":   "search",
        "ooh_S":      "ooh",
    }

    for col, name in channels.items():
        if col in df.columns:
            params = fit_adstock_params(
                df[col].values.astype(float),
                df["revenue"].values.astype(float),
                channel_name=name,
            )
            adstock_params[col] = params

            # Apply the transform — creates tv_S_transformed, etc.
            df = apply_adstock_and_saturation(
                df, params, raw_col=col, out_col=f"{col}_transformed"
            )

    # ── Lag features on raw spend ─────────────────────────────────────────
    for col in ["tv_S", "facebook_S", "search_S"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)

    # ── Rolling spend (captures sustained campaigns) ──────────────────────
    df["tv_rolling4"] = df["tv_S"].rolling(4).mean()
    df["fb_rolling4"] = df["facebook_S"].rolling(4).mean()

    # ── Share of voice ────────────────────────────────────────────────────
    total_spend = df[["tv_S", "facebook_S", "search_S", "ooh_S"]].sum(axis=1)
    df["tv_sov"] = df["tv_S"] / (total_spend + 1e-9)

    df = df.dropna().reset_index(drop=True)

    return df, adstock_params