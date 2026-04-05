import pandas as pd
import numpy as np
from scipy import stats

def simulate_geo_lift(df: pd.DataFrame,
                      treatment_fraction: float = 0.6,
                      tv_lift_effect: float = 0.12,
                      seed: int = 42) -> pd.DataFrame:
    """
    Simulate a geo-holdout experiment.

    In a real Tatari experiment:
    - 'treatment' DMAs continue receiving TV ads
    - 'holdout' DMAs have TV ads paused
    - Lift = (treatment conversions - holdout conversions) / holdout conversions

    We simulate 20 DMAs, split them, inject a TV effect in treatment.
    """
    rng = np.random.default_rng(seed)
    n_weeks = len(df)
    dmas = [f"DMA_{i:02d}" for i in range(20)]

    records = []
    for dma in dmas:
        is_treatment = rng.random() < treatment_fraction
        # Base revenue correlated with df but with DMA-level noise
        base_rev = df["revenue"].values * rng.uniform(0.7, 1.3)

        # Inject TV lift only in treatment DMAs
        tv_multiplier = (1 + tv_lift_effect) if is_treatment else 1.0
        obs_rev = base_rev * tv_multiplier * rng.uniform(0.92, 1.08, n_weeks)

        for i, (date, rev, base) in enumerate(zip(df["date"], obs_rev, base_rev)):
            records.append({
                "date": date,
                "dma": dma,
                "group": "treatment" if is_treatment else "holdout",
                "observed_revenue": rev,
                "baseline_revenue": base,
                "tv_spend": df["tv_S"].iloc[i] if is_treatment else 0.0,
            })

    return pd.DataFrame(records)


def measure_geo_lift(geo_df: pd.DataFrame) -> dict:
    """
    Measure incremental lift between treatment and holdout groups.
    This is the key causal measurement.
    """
    treatment = geo_df[geo_df["group"] == "treatment"]["observed_revenue"]
    holdout   = geo_df[geo_df["group"] == "holdout"]["observed_revenue"]

    mean_t = treatment.mean()
    mean_h = holdout.mean()
    lift_pct = (mean_t - mean_h) / mean_h * 100

    # Statistical significance (t-test)
    t_stat, p_value = stats.ttest_ind(treatment, holdout)

    # Incremental ROAS
    treatment_rows = geo_df[geo_df["group"] == "treatment"]
    total_tv_spend = treatment_rows["tv_spend"].sum()
    incremental_rev = (mean_t - mean_h) * len(treatment)
    iroas = incremental_rev / (total_tv_spend + 1e-9)

    return {
        "lift_pct": round(lift_pct, 2),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "incremental_roas": round(iroas, 3),
        "treatment_mean": round(mean_t, 2),
        "holdout_mean": round(mean_h, 2),
    }