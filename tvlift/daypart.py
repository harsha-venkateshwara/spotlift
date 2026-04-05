import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


DAYPARTS = ["morning", "daytime", "primetime", "late_night"]
DAYS     = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def simulate_airing_level_data(
    df: pd.DataFrame,
    n_airings_per_week: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate individual TV airing records with daypart and day-of-week.

    Ground truth ROAS multipliers are baked in so the model
    has a real signal to learn from. Primetime weekends are the
    most valuable slots — matching real-world TV buying intuition.
    """
    rng = np.random.default_rng(seed)

    # Ground truth ROAS by slot — model should recover this ranking
    slot_roas = {
        ("primetime",  "sat"): 2.8,
        ("primetime",  "sun"): 2.5,
        ("primetime",  "fri"): 2.2,
        ("primetime",  "mon"): 1.9,
        ("primetime",  "tue"): 1.8,
        ("primetime",  "wed"): 1.7,
        ("primetime",  "thu"): 1.8,
        ("daytime",    "sat"): 1.4,
        ("daytime",    "sun"): 1.3,
        ("daytime",    "mon"): 1.1,
        ("daytime",    "tue"): 1.0,
        ("daytime",    "wed"): 1.0,
        ("daytime",    "thu"): 1.1,
        ("daytime",    "fri"): 1.2,
        ("morning",    "sat"): 1.0,
        ("morning",    "sun"): 0.9,
        ("morning",    "mon"): 0.8,
        ("morning",    "tue"): 0.7,
        ("morning",    "wed"): 0.7,
        ("morning",    "thu"): 0.7,
        ("morning",    "fri"): 0.8,
        ("late_night", "sat"): 0.6,
        ("late_night", "sun"): 0.5,
        ("late_night", "mon"): 0.4,
        ("late_night", "tue"): 0.4,
        ("late_night", "wed"): 0.4,
        ("late_night", "thu"): 0.4,
        ("late_night", "fri"): 0.5,
    }

    # Base cost per airing by daypart (realistic CPM-based pricing)
    slot_cost_base = {
        "primetime":  30000,
        "daytime":    8000,
        "morning":    5000,
        "late_night": 3000,
    }

    records = []
    for _, row in df.iterrows():
        for _ in range(n_airings_per_week):
            daypart     = rng.choice(DAYPARTS, p=[0.15, 0.30, 0.40, 0.15])
            day         = rng.choice(DAYS)
            cost        = slot_cost_base[daypart] * rng.uniform(0.85, 1.15)
            impressions = cost * rng.uniform(800, 1200)

            true_roas   = slot_roas.get((daypart, day), 1.0)

            # Incremental conversions proportional to cost and true ROAS
            incr_conv   = (cost / 10000) * true_roas * rng.uniform(0.8, 1.2)

            records.append({
                "date":             row["date"],
                "daypart":          daypart,
                "day_of_week":      day,
                "cost":             cost,
                "impressions":      impressions,
                "incr_conversions": incr_conv,
                "true_roas":        true_roas,
                "week_of_year":     row["week_of_year"],
                "month":            row["month"],
            })

    return pd.DataFrame(records)


def train_daypart_model(airings_df: pd.DataFrame) -> tuple:
    """
    Train a GBM to predict incremental conversions per airing.
    One-hot encodes daypart and day_of_week, then trains on
    cost, impressions, and seasonality features.
    """
    df = airings_df.copy()
    df = pd.get_dummies(df, columns=["daypart", "day_of_week"], drop_first=False)

    exclude = {"date", "incr_conversions", "true_roas"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [float, int, bool, np.float64, np.int64, np.bool_]
    ]

    X = df[feature_cols].fillna(0)
    y = df["incr_conversions"]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    return model, feature_cols, float(cv_scores.mean())


def build_roas_heatmap(
    model,
    feature_cols: list,
    airings_df: pd.DataFrame,
    revenue_per_conversion: float = 5000.0,
) -> pd.DataFrame:
    """
    Build a daypart × day-of-week ROAS heatmap.

    For each slot combination, predict incremental conversions
    under a standard airing cost, then compute:
        ROAS = (predicted_conversions × revenue_per_conversion) / cost

    revenue_per_conversion = $5,000 — realistic average order value
    for DTC brands advertising on TV.
    """
    avg_cost        = airings_df["cost"].mean()
    avg_impressions = airings_df["impressions"].mean()
    avg_week        = airings_df["week_of_year"].mean()
    avg_month       = airings_df["month"].mean()

    rows = []
    for daypart in DAYPARTS:
        for day in DAYS:
            base = {col: 0 for col in feature_cols}
            base["cost"]          = avg_cost
            base["impressions"]   = avg_impressions
            base["week_of_year"]  = avg_week
            base["month"]         = avg_month

            dp_col  = f"daypart_{daypart}"
            day_col = f"day_of_week_{day}"
            if dp_col  in feature_cols: base[dp_col]  = 1
            if day_col in feature_cols: base[day_col] = 1

            X_pred    = pd.DataFrame([base])[feature_cols].fillna(0)
            pred_conv = float(model.predict(X_pred)[0])
            pred_rev  = pred_conv * revenue_per_conversion
            roas      = pred_rev / (avg_cost + 1e-9)

            rows.append({
                "daypart":        daypart,
                "day_of_week":    day,
                "predicted_roas": round(roas, 3),
            })

    return pd.DataFrame(rows)