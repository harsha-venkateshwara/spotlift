import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(__file__))

from tvlift.pipeline   import load_and_engineer
from tvlift.attribution import naive_attribution, ols_attribution, compute_incremental_roas
from tvlift.geo_lift   import simulate_geo_lift, measure_geo_lift
from tvlift.model      import train_xgb
from tvlift.optimizer  import optimize_budget

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TVLift · TV Attribution Engine",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "tv":       "#7F77DD",
    "facebook": "#1D9E75",
    "search":   "#D85A30",
    "ooh":      "#BA7517",
    "actual":   "#378ADD",
    "counter":  "#E24B4A",
}

# ── Load + cache ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return load_and_engineer("data/robyn_dt_simulated.csv")

@st.cache_resource
def load_model(df):
    return train_xgb(df)

@st.cache_data
def load_geo(df):
    geo_df = simulate_geo_lift(df)
    metrics = measure_geo_lift(geo_df)
    return geo_df, metrics

df = load_data()
model, feature_cols, cv_mape, shap_vals, X_train = load_model(df)
geo_df, geo_metrics = load_geo(df)
naive = naive_attribution(df)
ols_model, coefs, r2, scaler = ols_attribution(df)
iroas = compute_incremental_roas(df, coefs)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/TVLift-Tatari%20Project-7F77DD?style=for-the-badge")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🎯 Attribution", "📈 Geo-Lift", "💰 Optimizer"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Meta Robyn MMM (open source)")
st.sidebar.markdown("**Model:** XGBoost + BSTS causal inference")
st.sidebar.markdown(f"**CV MAPE:** {cv_mape:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("TVLift — TV Ad Incrementality Engine")
    st.caption("Built to mirror Tatari's convergent TV measurement methodology")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total TV Spend",     f"${df['tv_S'].sum():,.0f}")
    col2.metric("Total Revenue",       f"${df['revenue'].sum():,.0f}")
    col3.metric("TV Incremental ROAS", f"{iroas.get('tv_S', 0):.2f}x")
    col4.metric("Geo Lift (TV)",       f"{geo_metrics['lift_pct']:.1f}%",
                delta="Statistically significant" if geo_metrics['significant'] else "Not significant")

    # Spend over time
    st.subheader("Weekly spend by channel")
    fig = go.Figure()
    for col, label, color in [
        ("tv_S", "TV", COLORS["tv"]),
        ("facebook_S", "Facebook", COLORS["facebook"]),
        ("search_S", "Search", COLORS["search"]),
        ("ooh_S", "OOH", COLORS["ooh"]),
    ]:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col],
            name=label, line=dict(color=color, width=2),
            fill="tozeroy" if col == "tv_S" else None,
            fillcolor=f"rgba(127,119,221,0.08)" if col == "tv_S" else None,
        ))
    fig.update_layout(height=340, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # Revenue vs TV spend scatter
    st.subheader("TV spend vs revenue (with adstock lag)")
    fig2 = px.scatter(df, x="tv_S", y="revenue",
                      color="tv_rolling4",
                      color_continuous_scale="Purples",
                      labels={"tv_S": "TV Spend", "revenue": "Revenue",
                               "tv_rolling4": "4-wk rolling spend"},
                      trendline="ols")
    fig2.update_layout(height=320, template="plotly_dark",
                       paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Attribution":
    st.title("Attribution — Naive vs Causal")
    st.info("💡 **The core insight:** naive attribution over-credits the last channel touched. "
            "Causal models isolate what TV actually *caused*, not just correlated with.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Naive (proportional) attribution")
        labels = [k.replace("_S","").upper() for k in naive.keys()]
        values = list(naive.values())
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.45,
            marker_colors=[COLORS["tv"], COLORS["facebook"],
                           COLORS["search"], COLORS["ooh"], "#888"]
        ))
        fig.update_layout(height=300, template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("TV gets credit proportional to spend — ignores timing, incrementality, and causation.")

    with col2:
        st.subheader("Incremental ROAS (causal, OLS baseline)")
        channels = list(iroas.keys())
        values_roas = list(iroas.values())
        fig2 = go.Figure(go.Bar(
            x=[c.replace("_S","").upper() for c in channels],
            y=values_roas,
            marker_color=[COLORS["tv"], COLORS["facebook"], COLORS["search"]],
            text=[f"{v:.2f}x" for v in values_roas],
            textposition="outside"
        ))
        fig2.update_layout(height=300, template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           yaxis_title="Incremental ROAS")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Incremental ROAS = revenue *caused* by spend / spend. TV often looks weaker here — that's the point.")

    # Feature importance (SHAP)
    st.subheader("XGBoost feature importance (SHAP)")
    import shap
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_cols, "importance": mean_shap})
    shap_df = shap_df.sort_values("importance", ascending=True)
    fig3 = go.Figure(go.Bar(
        x=shap_df["importance"], y=shap_df["feature"],
        orientation="h",
        marker_color=COLORS["tv"]
    ))
    fig3.update_layout(height=350, template="plotly_dark",
                       paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GEO LIFT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Geo-Lift":
    st.title("Geo-Lift — Causal TV Measurement")
    st.info("💡 **Geo-lift methodology:** pause TV in holdout DMAs, keep it running in treatment DMAs. "
            "The revenue gap = TV's incremental impact. This is what Tatari does for every client.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Measured Lift", f"{geo_metrics['lift_pct']:.1f}%")
    col2.metric("Incremental ROAS", f"{geo_metrics['incremental_roas']:.2f}x")
    col3.metric("p-value", f"{geo_metrics['p_value']:.4f}",
                delta="Significant ✓" if geo_metrics['significant'] else "Not significant")

    # Treatment vs holdout over time
    st.subheader("Treatment vs holdout revenue over time")
    weekly = geo_df.groupby(["date", "group"])["observed_revenue"].mean().reset_index()
    fig = go.Figure()
    for group, color in [("treatment", COLORS["actual"]), ("holdout", COLORS["counter"])]:
        d = weekly[weekly["group"] == group]
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["observed_revenue"],
            name=group.capitalize(),
            line=dict(color=color, width=2.5)
        ))
    fig.update_layout(height=340, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      yaxis_title="Avg weekly revenue",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # Distribution comparison
    st.subheader("Revenue distribution: treatment vs holdout")
    fig2 = go.Figure()
    for group, color in [("treatment", COLORS["actual"]), ("holdout", COLORS["counter"])]:
        vals = geo_df[geo_df["group"] == group]["observed_revenue"]
        fig2.add_trace(go.Histogram(
            x=vals, name=group.capitalize(),
            opacity=0.7, marker_color=color, nbinsx=40
        ))
    fig2.update_layout(barmode="overlay", height=300,
                       template="plotly_dark",
                       paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Optimizer":
    st.title("Budget Optimizer — Maximize Incremental ROAS")
    st.info("💡 Use the slider to set your total budget. The optimizer finds the spend allocation "
            "that maximizes predicted incremental revenue across TV, Facebook, and Search.")

    budget = st.slider(
        "Total weekly budget ($)",
        min_value=10_000, max_value=500_000,
        value=100_000, step=5_000,
        format="$%d"
    )

    with st.spinner("Optimizing allocation..."):
        result = optimize_budget(model, feature_cols, df, total_budget=budget)

    alloc = result["optimal_allocation"]
    opt_rev = result["optimal_predicted_revenue"]
    eq_rev  = result["equal_split_revenue"]
    uplift  = result["revenue_uplift_pct"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Revenue (optimal)", f"${opt_rev:,.0f}")
    col2.metric("vs Equal Split",              f"${eq_rev:,.0f}")
    col3.metric("Revenue Uplift",              f"{uplift:.1f}%",
                delta=f"+{uplift:.1f}% from optimization")

    # Allocation bar chart
    st.subheader("Optimal spend allocation")
    channels = list(alloc.keys())
    values   = list(alloc.values())
    colors   = [COLORS["tv"], COLORS["facebook"], COLORS["search"]]
    labels   = ["TV", "Facebook", "Search"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"${v:,.0f}\n({v/budget*100:.0f}%)" for v in values],
        textposition="outside"
    ))
    fig.add_hline(y=budget/3, line_dash="dot",
                  annotation_text="Equal split baseline",
                  line_color="gray")
    fig.update_layout(height=340, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      yaxis_title="Spend ($)")
    st.plotly_chart(fig, use_container_width=True)

    # ROAS heatmap across TV/FB allocations
    st.subheader("ROAS landscape — TV share vs Facebook share")
    all_res = result["all_results"]
    if not all_res.empty:
        all_res["tv_share_pct"] = (all_res["tv_S"] / budget * 100).round(0)
        all_res["fb_share_pct"] = (all_res["facebook_S"] / budget * 100).round(0)
        pivot = all_res.pivot_table(
            index="fb_share_pct", columns="tv_share_pct",
            values="predicted_revenue", aggfunc="max"
        )
        fig2 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale="Purples",
            colorbar=dict(title="Pred. revenue")
        ))
        fig2.update_layout(height=340, template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="TV share (%)",
                           yaxis_title="Facebook share (%)")
        st.plotly_chart(fig2, use_container_width=True)