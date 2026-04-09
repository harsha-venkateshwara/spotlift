import numpy as np
import pandas as pd


def optimize_budget(
    model,
    feature_cols: list,
    df: pd.DataFrame,
    total_budget: float = 100_000,
    channel_bounds: dict = None,
) -> dict:
    """
    Budget optimizer that works correctly across any budget level.
    Core insight: XGBoost was trained on normalized/transformed features.
    We must feed it inputs in the same scale it saw during training.
    Strategy:
      - Compute the historical spend distribution per channel
      - Express the new budget as a MULTIPLIER on historical means
      - Apply that multiplier to all spend-derived features
      - This keeps the model in-distribution regardless of budget size
    """
    channels = ["tv_S", "facebook_S", "search_S"]

    if channel_bounds is None:
        channel_bounds = {
            "tv_S":       (0.10, 0.70),
            "facebook_S": (0.10, 0.50),
            "search_S":   (0.10, 0.50),
        }

    tv_min, tv_max = channel_bounds.get("tv_S",       (0.10, 0.70))
    fb_min, fb_max = channel_bounds.get("facebook_S", (0.10, 0.50))
    sr_min, sr_max = channel_bounds.get("search_S",   (0.10, 0.50))

    # Historical means — this is the scale the model was trained on
    hist_means = {ch: df[ch].mean() for ch in channels}
    hist_total = sum(hist_means.values())

    # Base feature row: historical means for everything
    base_row = df[feature_cols].mean().to_dict()

    def build_feature_row(tv_share: float, fb_share: float, sr_share: float) -> np.ndarray:
        """
        Build a model-ready feature vector for a given allocation.
        We thenscale each channel's features by the ratio of:
            (new spend) / (historical mean spend)

        This preserves relative magnitudes the model learned during training.
        """
        alloc = {
            "tv_S":       tv_share * total_budget,
            "facebook_S": fb_share * total_budget,
            "search_S":   sr_share * total_budget,
        }

        row = base_row.copy()

        for ch in channels:
            new_spend = alloc[ch]
            hist_mean = hist_means[ch] + 1e-9

            # Scale factor: how much more/less than historical average
            scale = new_spend / hist_mean

            # Raw spend feature
            if ch in feature_cols:
                row[ch] = new_spend

            # Adstock-transformed feature scales with spend
            transformed_col = f"{ch}_transformed"
            if transformed_col in feature_cols:
                row[transformed_col] = base_row[transformed_col] * scale

            # Lag features scale with spend too
            for lag in ["_lag1", "_lag2"]:
                lag_col = f"{ch}{lag}"
                if lag_col in feature_cols:
                    row[lag_col] = base_row[lag_col] * scale

        # Rolling average for TV
        if "tv_rolling4" in feature_cols:
            tv_scale = alloc["tv_S"] / (hist_means["tv_S"] + 1e-9)
            row["tv_rolling4"] = base_row["tv_rolling4"] * tv_scale

        # Share of voice
        new_total = sum(alloc.values())
        if "tv_sov" in feature_cols:
            row["tv_sov"] = alloc["tv_S"] / (new_total + 1e-9)

        # Seasonality features stay at their mean — not spend-dependent
        for col in ["week_of_year", "month", "sin_week", "cos_week"]:
            if col in feature_cols:
                row[col] = base_row[col]

        return np.array([[row.get(f, 0) for f in feature_cols]])

    #  Grid search over allocation fractions
    best_rev   = -np.inf
    best_alloc = None
    results    = []

    step = 0.05
    for tv_share in np.arange(tv_min, tv_max + 0.001, step):
        for fb_share in np.arange(fb_min, fb_max + 0.001, step):
            sr_share = round(1.0 - tv_share - fb_share, 6)

            if sr_share < sr_min - 0.001 or sr_share > sr_max + 0.001:
                continue

            X_pred   = build_feature_row(tv_share, fb_share, sr_share)
            pred_rev = float(model.predict(X_pred)[0])

            alloc = {
                "tv_S":       tv_share * total_budget,
                "facebook_S": fb_share * total_budget,
                "search_S":   sr_share * total_budget,
            }

            results.append({
                **alloc,
                "tv_share":          round(tv_share, 3),
                "fb_share":          round(fb_share, 3),
                "sr_share":          round(sr_share, 3),
                "predicted_revenue": pred_rev,
            })

            if pred_rev > best_rev:
                best_rev   = pred_rev
                best_alloc = alloc.copy()

    # Equal-split baseline 
    eq_share  = 1 / 3
    X_equal   = build_feature_row(eq_share, eq_share, eq_share)
    equal_rev = float(model.predict(X_equal)[0])

    # Uplift relative to equal split
    uplift_pct = (best_rev - equal_rev) / (abs(equal_rev) + 1e-9) * 100

    return {
        "optimal_allocation":        best_alloc,
        "optimal_predicted_revenue": round(best_rev, 2),
        "equal_split_revenue":       round(equal_rev, 2),
        "revenue_uplift_pct":        round(uplift_pct, 2),
        "all_results":               pd.DataFrame(results),
    }