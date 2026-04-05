import numpy as np
import pandas as pd
from scipy.optimize import minimize

def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Geometric adstock transform.
    theta = decay rate (0 = no carryover, 1 = infinite carryover)
    TV typically: 0.3–0.7. Search: 0.0–0.2.

    At each time t:
        adstock[t] = x[t] + theta * adstock[t-1]

    This captures the 'memory' of TV — an airing this week
    still drives some conversions next week and the week after.
    """
    adstock = np.zeros_like(x, dtype=float)
    adstock[0] = x[0]
    for t in range(1, len(x)):
        adstock[t] = x[t] + theta * adstock[t - 1]
    return adstock


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill transformation for diminishing returns.

    alpha = shape of the S-curve (alpha > 1 = S-shaped, alpha < 1 = concave)
    gamma = half-saturation point (spend level where you get 50% of max response)

    This prevents the optimizer from naively saying
    'just pour infinite money into TV' — diminishing returns are real.
    """
    x = np.clip(x, 0, None)
    return x ** alpha / (x ** alpha + gamma ** alpha)


def fit_adstock_params(
    spend: np.ndarray,
    revenue: np.ndarray,
    channel_name: str = "tv"
) -> dict:
    """
    Fit adstock theta and Hill (alpha, gamma) jointly by minimizing
    residual sum of squares between transformed spend and revenue.

    Returns a dict of best-fit parameters.
    This is a simplified single-channel fit — in a full MMM you'd
    fit all channels simultaneously inside the Bayesian model.
    """
    def loss(params):
        theta, alpha, gamma = params
        theta = np.clip(theta, 0.01, 0.99)
        alpha = np.clip(alpha, 0.1, 5.0)
        gamma = np.clip(gamma, 1e-3, np.percentile(spend, 90))

        transformed = hill_saturation(
            geometric_adstock(spend, theta), alpha, gamma
        )
        # simple OLS loss
        coef = np.dot(transformed, revenue) / (np.dot(transformed, transformed) + 1e-9)
        resid = revenue - coef * transformed
        return np.sum(resid ** 2)

    # reasonable starting values
    x0 = [0.4, 1.0, np.median(spend[spend > 0])]
    bounds = [(0.01, 0.99), (0.1, 5.0), (1e-3, np.percentile(spend[spend > 0], 90))]

    result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    theta, alpha, gamma = result.x

    return {
        "channel": channel_name,
        "theta": round(float(theta), 4),
        "alpha": round(float(alpha), 4),
        "gamma": round(float(gamma), 4),
        "converged": result.success,
        "loss": round(float(result.fun), 2),
    }


def apply_adstock_and_saturation(
    df: pd.DataFrame,
    params: dict,
    raw_col: str,
    out_col: str
) -> pd.DataFrame:
    """
    Apply fitted adstock + Hill transforms to a spend column.
    Creates a new column ready to drop into the model.
    """
    raw = df[raw_col].values.astype(float)
    adstocked = geometric_adstock(raw, params["theta"])
    transformed = hill_saturation(adstocked, params["alpha"], params["gamma"])
    df[out_col] = transformed
    return df


def build_response_curve(
    params: dict,
    max_spend: float,
    n_points: int = 200
) -> pd.DataFrame:
    """
    Build a spend → response curve for visualization.
    Shows diminishing returns clearly.
    """
    spend_grid = np.linspace(0, max_spend, n_points)
    # single-period (no adstock needed for the curve shape illustration)
    response = hill_saturation(spend_grid, params["alpha"], params["gamma"])

    return pd.DataFrame({
        "spend": spend_grid,
        "response": response,
        "marginal_response": np.gradient(response, spend_grid),
    })