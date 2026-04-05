import numpy as np
import pandas as pd
from scipy import stats


def simulate_geo_lift(
    df: pd.DataFrame,
    treatment_fraction: float = 0.6,
    tv_lift_effect: float = 0.12,
    n_dmas: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a geo-holdout experiment across DMAs.

    Real TV Ad Company workflow:
      1. Select holdout DMAs before the campaign starts
      2. Run TV only in treatment DMAs
      3. Measure revenue gap after 4-8 weeks
      4. Divide incremental revenue by TV spend = iROAS

    We inject a known ground truth lift so we can validate
    our estimation method recovers it accurately.
    """
    rng    = np.random.default_rng(seed)
    n_weeks = len(df)
    dmas   = [f"DMA_{i:02d}" for i in range(n_dmas)]

    records = []
    for dma in dmas:
        is_treatment = rng.random() < treatment_fraction
        dma_scale    = rng.uniform(0.5, 1.8)
        base_rev     = df["revenue"].values * dma_scale

        tv_multiplier = (1 + tv_lift_effect) if is_treatment else 1.0
        noise         = rng.uniform(0.90, 1.10, n_weeks)
        obs_rev       = base_rev * tv_multiplier * noise

        for i in range(n_weeks):
            records.append({
                "date":             df["date"].iloc[i],
                "dma":              dma,
                "group":            "treatment" if is_treatment else "holdout",
                "observed_revenue": obs_rev[i],
                "baseline_revenue": base_rev[i],
                "tv_spend":         df["tv_S"].iloc[i] * dma_scale if is_treatment else 0.0,
                "dma_scale":        dma_scale,
            })

    return pd.DataFrame(records)


def bootstrap_lift(
    geo_df: pd.DataFrame,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals on geo-lift.

    Instead of trusting a single DMA split, we resample DMA
    assignments repeatedly to quantify estimation uncertainty.

    This answers: "How much would the lift estimate change if
    we'd picked slightly different holdout DMAs?"

    A 95% CI of [8%, 17%] tells a client something real.
    A point estimate of 12% tells them almost nothing.
    """
    rng = np.random.default_rng(seed)

    treatment_dmas = geo_df[geo_df["group"] == "treatment"]["dma"].unique()
    holdout_dmas   = geo_df[geo_df["group"] == "holdout"]["dma"].unique()

    lift_samples  = []
    iroas_samples = []

    for _ in range(n_iterations):
        sampled_t = rng.choice(treatment_dmas, size=len(treatment_dmas), replace=True)
        sampled_h = rng.choice(holdout_dmas,   size=len(holdout_dmas),   replace=True)

        t_rev = geo_df[geo_df["dma"].isin(sampled_t)]["observed_revenue"].mean()
        h_rev = geo_df[geo_df["dma"].isin(sampled_h)]["observed_revenue"].mean()

        lift = (t_rev - h_rev) / (h_rev + 1e-9) * 100
        lift_samples.append(lift)

        t_spend  = geo_df[geo_df["dma"].isin(sampled_t)]["tv_spend"].sum()
        t_count  = len(sampled_t)
        incr_rev = (t_rev - h_rev) * t_count
        iroas    = incr_rev / (t_spend + 1e-9)
        iroas_samples.append(iroas)

    alpha     = 1 - confidence
    lift_arr  = np.array(lift_samples)
    iroas_arr = np.array(iroas_samples)

    t_rev_obs = geo_df[geo_df["group"] == "treatment"]["observed_revenue"]
    h_rev_obs = geo_df[geo_df["group"] == "holdout"]["observed_revenue"]
    _, p_value = stats.ttest_ind(t_rev_obs, h_rev_obs)

    return {
        "lift_pct":           round(float(np.median(lift_arr)), 2),
        "incremental_roas":   round(float(np.median(iroas_arr)), 3),
        "lift_ci_low":        round(float(np.percentile(lift_arr,  alpha / 2 * 100)), 2),
        "lift_ci_high":       round(float(np.percentile(lift_arr,  (1 - alpha / 2) * 100)), 2),
        "iroas_ci_low":       round(float(np.percentile(iroas_arr, alpha / 2 * 100)), 3),
        "iroas_ci_high":      round(float(np.percentile(iroas_arr, (1 - alpha / 2) * 100)), 3),
        "lift_distribution":  lift_arr.tolist(),
        "iroas_distribution": iroas_arr.tolist(),
        "p_value":            round(float(p_value), 4),
        "significant":        bool(p_value < 0.05),
        "confidence_level":   confidence,
        "n_iterations":       n_iterations,
    }