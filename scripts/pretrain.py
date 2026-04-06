import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
from tvlift.pipeline import load_and_engineer
from tvlift.model    import train_xgb
from tvlift.geo_lift import simulate_geo_lift, bootstrap_lift
from tvlift.daypart  import (
    simulate_airing_level_data,
    train_daypart_model,
    build_roas_heatmap,
)
from tvlift.power_analysis import run_full_power_analysis

print("Loading data...")
df, adstock_params = load_and_engineer("data/robyn_dt_simulated.csv")

print("Training XGBoost model...")
model, feature_cols, cv_mape, shap_vals, X_train = train_xgb(df)

print("Running geo-lift simulation...")
geo_df  = simulate_geo_lift(df)
metrics = bootstrap_lift(geo_df, n_iterations=1000)

print("Running power analysis...")
power_df = run_full_power_analysis(geo_df, target_lift_pct=10.0)

print("Training daypart model...")
airings_df = simulate_airing_level_data(df)
dp_model, dp_feature_cols, dp_r2 = train_daypart_model(airings_df)
heatmap_df = build_roas_heatmap(dp_model, dp_feature_cols, airings_df)

print("Saving all artifacts...")
os.makedirs("artifacts", exist_ok=True)

# Saving everything
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "cv_mape":      cv_mape,
        "shap_vals":    shap_vals,
        "X_train":      X_train,
    }, f)

with open("artifacts/adstock_params.pkl", "wb") as f:
    pickle.dump(adstock_params, f)

with open("artifacts/geo.pkl", "wb") as f:
    pickle.dump({
        "geo_df":  geo_df,
        "metrics": metrics,
    }, f)

with open("artifacts/power.pkl", "wb") as f:
    pickle.dump(power_df, f)

with open("artifacts/daypart.pkl", "wb") as f:
    pickle.dump({
        "airings_df": airings_df,
        "heatmap_df": heatmap_df,
        "r2":         dp_r2,
    }, f)

import pandas as pd
df.to_csv("artifacts/df_engineered.csv", index=False)

print("Done! All artifacts saved to artifacts/")
print(f"CV MAPE: {cv_mape:.1%}")
print(f"Geo lift: {metrics['lift_pct']}%")
print(f"Daypart R2: {dp_r2:.2f}")