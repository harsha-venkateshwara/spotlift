import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from typing import Optional


def build_bayesian_mmm(
    df: pd.DataFrame,
    adstock_params: dict,
    sample_kwargs: Optional[dict] = None,
):
    """
    Bayesian Media Mix Model using PyMC.

    This is the production-grade approach used by TV Ad platform companies like Meta,
    Google, and every serious MMM team. Instead of a point estimate
    ("TV ROAS = 1.8x"), you get a full posterior distribution
    ("TV ROAS is 1.8x, 94% HDI: 1.1x – 2.7x").

    Model structure:
      revenue ~ Normal(mu, sigma)
      mu = baseline + sum(beta_c * adstock_hill(spend_c)) + seasonality

    Priors encode domain knowledge:
      - TV typically has positive but uncertain ROAS
      - Adstock decay is bounded between 0 and 1
      - We expect diminishing returns (Hill alpha < 2)
    """
    if sample_kwargs is None:
        sample_kwargs = {
            "draws": 1000,
            "tune": 500,
            "chains": 2,
            "target_accept": 0.9,
            "return_inferencedata": True,
            "progressbar": True,
        }

    channels = ["tv_S", "facebook_S", "search_S", "ooh_S"]
    n = len(df)

    # Pre-apply adstock (we fit theta outside Bayesian model for speed)
    # In a full production MMM you'd put theta inside the model too
    from tvlift.adstock import geometric_adstock, hill_saturation

    spend_arrays = {}
    for ch in channels:
        raw = df[ch].values.astype(float)
        theta = adstock_params.get(ch, {}).get("theta", 0.3)
        spend_arrays[ch] = geometric_adstock(raw, theta)

    # Standardize revenue for numerical stability
    rev = df["revenue"].values.astype(float)
    rev_mean = rev.mean()
    rev_std  = rev.std()
    rev_scaled = (rev - rev_mean) / rev_std

    # Seasonality: week-of-year sine/cosine encoding
    week = df["week_of_year"].values
    sin_w = np.sin(2 * np.pi * week / 52)
    cos_w = np.cos(2 * np.pi * week / 52)

    with pm.Model() as mmm:

        # ── Channel betas (contribution to revenue) ──────────────────────
        # Half-normal prior: we believe channels contribute positively
        # but don't want to assume how much
        beta_tv       = pm.HalfNormal("beta_tv",       sigma=1.0)
        beta_facebook = pm.HalfNormal("beta_facebook",  sigma=1.0)
        beta_search   = pm.HalfNormal("beta_search",    sigma=1.0)
        beta_ooh      = pm.HalfNormal("beta_ooh",       sigma=0.5)

        # ── Hill saturation params per channel ───────────────────────────
        # alpha: shape. Prior centered on 1 (concave) — most spend-response
        # relationships are concave or mildly S-shaped, not strongly S-shaped
        alpha_tv = pm.Beta("alpha_tv", alpha=2, beta=2)  # ~ 0.5, bounded 0-1

        # gamma: half-saturation. Prior = 50th percentile of spend
        # meaning we expect the channel to half-saturate around median spend
        gamma_tv = pm.HalfNormal(
            "gamma_tv",
            sigma=float(np.percentile(spend_arrays["tv_S"][spend_arrays["tv_S"] > 0], 50))
        )

        # ── Seasonality ──────────────────────────────────────────────────
        beta_sin = pm.Normal("beta_sin", mu=0, sigma=0.5)
        beta_cos = pm.Normal("beta_cos", mu=0, sigma=0.5)

        # ── Baseline intercept ───────────────────────────────────────────
        intercept = pm.Normal("intercept", mu=0, sigma=1.0)

        # ── Noise ────────────────────────────────────────────────────────
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        # ── Apply Hill saturation to adstocked TV spend ──────────────────
        tv_adstocked = pt.as_tensor_variable(spend_arrays["tv_S"])
        tv_transformed = tv_adstocked ** alpha_tv / (
            tv_adstocked ** alpha_tv + gamma_tv ** alpha_tv + 1e-9
        )

        # ── Normalize other channels simply (no Hill for speed) ───────────
        fb_norm  = pt.as_tensor_variable(spend_arrays["facebook_S"] / (spend_arrays["facebook_S"].max() + 1e-9))
        sr_norm  = pt.as_tensor_variable(spend_arrays["search_S"]   / (spend_arrays["search_S"].max()   + 1e-9))
        ooh_norm = pt.as_tensor_variable(spend_arrays["ooh_S"]       / (spend_arrays["ooh_S"].max()       + 1e-9))
        sin_t    = pt.as_tensor_variable(sin_w)
        cos_t    = pt.as_tensor_variable(cos_w)

        # ── Expected revenue (scaled) ────────────────────────────────────
        mu = (
            intercept
            + beta_tv       * tv_transformed
            + beta_facebook * fb_norm
            + beta_search   * sr_norm
            + beta_ooh      * ooh_norm
            + beta_sin      * sin_t
            + beta_cos      * cos_t
        )

        # ── Likelihood ───────────────────────────────────────────────────
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=rev_scaled)

        # ── Sample ───────────────────────────────────────────────────────
        trace = pm.sample(**sample_kwargs)

    return mmm, trace, rev_mean, rev_std, spend_arrays


def extract_channel_roas(
    trace,
    df: pd.DataFrame,
    spend_arrays: dict,
    rev_mean: float,
    rev_std: float,
) -> pd.DataFrame:
    """
    Extract posterior ROAS distribution for each channel.

    ROAS = (revenue attributable to channel) / (spend on channel)

    We compute this for every posterior sample, giving us
    a full distribution rather than a point estimate.
    """
    results = []
    channels = {
        "TV":       ("beta_tv",       "tv_S",       spend_arrays["tv_S"]),
        "Facebook": ("beta_facebook",  "facebook_S", spend_arrays["facebook_S"]),
        "Search":   ("beta_search",    "search_S",   spend_arrays["search_S"]),
        "OOH":      ("beta_ooh",       "ooh_S",      spend_arrays["ooh_S"]),
    }

    for channel_name, (beta_name, spend_col, spend_arr) in channels.items():
        if beta_name not in trace.posterior:
            continue

        betas = trace.posterior[beta_name].values.flatten()
        total_spend = df[spend_col].sum()

        if total_spend < 1:
            continue

        # Revenue attribution: beta * mean(transformed_spend) * n_weeks * scale
        mean_spend = spend_arr.mean()
        max_spend  = spend_arr.max() + 1e-9
        spend_norm = mean_spend / max_spend

        # Incremental revenue per unit of normalized spend, unscaled
        incr_rev_samples = betas * spend_norm * rev_std * len(df)
        roas_samples = incr_rev_samples / (total_spend + 1e-9)

        results.append({
            "channel": channel_name,
            "roas_median":  round(float(np.median(roas_samples)), 3),
            "roas_mean":    round(float(np.mean(roas_samples)), 3),
            "roas_hdi_low": round(float(np.percentile(roas_samples, 3)), 3),
            "roas_hdi_high":round(float(np.percentile(roas_samples, 97)), 3),
            "roas_samples": roas_samples.tolist(),
        })

    return pd.DataFrame(results)


def compute_counterfactual(
    trace,
    df: pd.DataFrame,
    spend_arrays: dict,
    rev_mean: float,
    rev_std: float,
) -> pd.DataFrame:
    """
    Counterfactual: what would revenue look like with zero TV spend?

    For each posterior sample, set TV spend to 0 and recompute mu.
    The gap between actual and counterfactual is TV's causal contribution.

    With uncertainty: the gap has a distribution — we can report
    "TV drove $X revenue (94% HDI: $Y – $Z)"
    """
    from tvlift.adstock import geometric_adstock

    betas_tv   = trace.posterior["beta_tv"].values.flatten()
    alpha_tv_s = trace.posterior["alpha_tv"].values.flatten()
    gamma_tv_s = trace.posterior["gamma_tv"].values.flatten()
    beta_sin_s = trace.posterior["beta_sin"].values.flatten()
    beta_cos_s = trace.posterior["beta_cos"].values.flatten()
    intercepts = trace.posterior["intercept"].values.flatten()

    week    = df["week_of_year"].values
    sin_w   = np.sin(2 * np.pi * week / 52)
    cos_w   = np.cos(2 * np.pi * week / 52)
    tv_ads  = spend_arrays["tv_S"]

    n_samples = min(500, len(betas_tv))  # cap for speed
    idx = np.random.choice(len(betas_tv), n_samples, replace=False)

    # Actual model predictions (with TV)
    actual_samples = []
    counter_samples = []

    for i in idx:
        b_tv    = betas_tv[i]
        a_tv    = alpha_tv_s[i]
        g_tv    = gamma_tv_s[i]
        b_sin   = beta_sin_s[i]
        b_cos   = beta_cos_s[i]
        intcpt  = intercepts[i]

        # With TV
        tv_t = tv_ads ** a_tv / (tv_ads ** a_tv + g_tv ** a_tv + 1e-9)
        mu_with = intcpt + b_tv * tv_t + b_sin * sin_w + b_cos * cos_w
        actual_samples.append(mu_with * rev_std + rev_mean)

        # Without TV (set TV contribution to 0)
        mu_without = intcpt + b_sin * sin_w + b_cos * cos_w
        counter_samples.append(mu_without * rev_std + rev_mean)

    actual_arr  = np.array(actual_samples)   # (n_samples, n_weeks)
    counter_arr = np.array(counter_samples)

    result = pd.DataFrame({
        "date":              df["date"].values,
        "actual_median":     np.median(actual_arr, axis=0),
        "actual_hdi_low":    np.percentile(actual_arr, 3, axis=0),
        "actual_hdi_high":   np.percentile(actual_arr, 97, axis=0),
        "counter_median":    np.median(counter_arr, axis=0),
        "counter_hdi_low":   np.percentile(counter_arr, 3, axis=0),
        "counter_hdi_high":  np.percentile(counter_arr, 97, axis=0),
        "tv_contribution":   np.median(actual_arr - counter_arr, axis=0),
    })

    return result