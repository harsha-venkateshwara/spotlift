import numpy as np
import pandas as pd
from scipy import stats


def compute_mde(
    n_dmas_per_group: int,
    baseline_revenue: float,
    revenue_std: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Minimum Detectable Effect (MDE) for a geo-lift experiment.

    This answers: "Given N DMAs per group, what's the smallest
    true lift we can reliably detect?"

    If your MDE is 20% but you only expect a 5% TV lift,
    you need more DMAs or a longer experiment — full stop.

    z_alpha: one-sided critical value for false positive rate
    z_beta:  critical value for desired power (1 - false negative rate)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-sided
    z_beta  = stats.norm.ppf(power)

    # Standard error of difference in means
    se = revenue_std * np.sqrt(2 / n_dmas_per_group)

    # MDE as absolute revenue lift
    mde_absolute = (z_alpha + z_beta) * se

    # Convert to percentage lift relative to baseline
    mde_pct = mde_absolute / baseline_revenue * 100

    return round(mde_pct, 2)


def compute_required_dmas(
    target_mde_pct: float,
    baseline_revenue: float,
    revenue_std: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Minimum DMAs per group to detect a given lift percentage.

    Use this before designing the experiment:
    "We expect TV to lift revenue ~10%. How many holdout DMAs do we need?"
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    target_absolute = target_mde_pct / 100 * baseline_revenue
    n = 2 * ((z_alpha + z_beta) * revenue_std / target_absolute) ** 2

    return int(np.ceil(n))


def compute_required_weeks(
    n_dmas_per_group: int,
    target_mde_pct: float,
    weekly_revenue_by_dma: pd.DataFrame,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Minimum experiment duration in weeks.

    More weeks = more data per DMA = lower variance = detectable smaller effects.
    This is the experiment duration vs sensitivity tradeoff.
    """
    results = []
    for n_weeks in range(1, 53):
        # Aggregate revenue per DMA over n_weeks
        sample = (
            weekly_revenue_by_dma
            .groupby("dma")["observed_revenue"]
            .apply(lambda x: x.iloc[:n_weeks].sum())
        )
        mu  = sample.mean()
        std = sample.std()

        mde = compute_mde(n_dmas_per_group, mu, std, alpha, power)
        results.append({
            "weeks": n_weeks,
            "mde_pct": mde,
            "detects_target": mde <= target_mde_pct,
        })

    results_df = pd.DataFrame(results)
    detects = results_df[results_df["detects_target"]]

    return {
        "curve": results_df,
        "min_weeks_needed": int(detects["weeks"].min()) if len(detects) else None,
        "target_mde_pct": target_mde_pct,
        "n_dmas_per_group": n_dmas_per_group,
    }


def run_full_power_analysis(
    geo_df: pd.DataFrame,
    dma_range: list = None,
    target_lift_pct: float = 10.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """
    Sweep over different DMA counts and compute MDE for each.
    Returns a table suitable for plotting the power curve.
    """
    if dma_range is None:
        dma_range = list(range(2, 31))

    baseline = geo_df["observed_revenue"].mean()
    std      = (
        geo_df.groupby("dma")["observed_revenue"]
        .mean()
        .std()
    )

    rows = []
    for n in dma_range:
        mde = compute_mde(n, baseline, std, alpha, power)
        rows.append({
            "dmas_per_group": n,
            "total_dmas": n * 2,
            "mde_pct": mde,
            "detects_target": mde <= target_lift_pct,
        })

    return pd.DataFrame(rows)