import numpy as np
import cvxpy as cp
import pandas as pd

def optimize_budget(model,
                    feature_cols: list,
                    df: pd.DataFrame,
                    total_budget: float = 100_000,
                    channel_bounds: dict = None) -> dict:
    """
    Given a total budget, find the optimal spend allocation across
    TV, Facebook, and Search to maximize predicted revenue.

    Uses cvxpy for convex optimization.
    The key insight for the interview: this is why Tatari measures
    incrementality — you can only optimize what you can measure.
    """
    channels = ["tv_S", "facebook_S", "search_S"]

    if channel_bounds is None:
        channel_bounds = {
            "tv_S":         (0.10, 0.70),  # TV: 10-70% of budget
            "facebook_S":   (0.10, 0.50),
            "search_S":     (0.10, 0.50),
        }

    # Baseline feature means (hold non-spend features constant)
    base_row = df[feature_cols].mean().to_dict()

    # Grid search over allocations (discretize for interpretability)
    best_rev = -np.inf
    best_alloc = None
    results = []

    step = 0.05
    for tv_share in np.arange(0.10, 0.71, step):
        for fb_share in np.arange(0.10, min(0.71, 1-tv_share), step):
            sr_share = 1.0 - tv_share - fb_share
            if sr_share < 0.10:
                continue

            alloc = {
                "tv_S":       tv_share * total_budget,
                "facebook_S": fb_share * total_budget,
                "search_S":   sr_share * total_budget,
            }

            row = base_row.copy()
            for ch, val in alloc.items():
                row[ch] = val
            row["tv_sov"] = alloc["tv_S"] / total_budget

            X_pred = np.array([[row[f] for f in feature_cols]])
            pred_rev = model.predict(X_pred)[0]

            results.append({**alloc, "predicted_revenue": pred_rev})
            if pred_rev > best_rev:
                best_rev = pred_rev
                best_alloc = alloc.copy()

    # Baseline: equal split
    equal_row = base_row.copy()
    for ch in channels:
        equal_row[ch] = total_budget / 3
    equal_row["tv_sov"] = 1/3
    X_equal = np.array([[equal_row[f] for f in feature_cols]])
    equal_rev = model.predict(X_equal)[0]

    return {
        "optimal_allocation": best_alloc,
        "optimal_predicted_revenue": round(best_rev, 2),
        "equal_split_revenue": round(equal_rev, 2),
        "revenue_uplift_pct": round((best_rev - equal_rev) / equal_rev * 100, 2),
        "all_results": pd.DataFrame(results),
    }