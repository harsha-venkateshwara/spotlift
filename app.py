import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.append(os.path.dirname(__file__))

from tvlift.pipeline       import load_and_engineer
from tvlift.attribution    import naive_attribution, ols_attribution, compute_incremental_roas
from tvlift.geo_lift       import simulate_geo_lift, bootstrap_lift
from tvlift.power_analysis import run_full_power_analysis
from tvlift.model          import train_xgb
from tvlift.optimizer      import optimize_budget
from tvlift.adstock        import build_response_curve
from tvlift.daypart        import (
    simulate_airing_level_data,
    train_daypart_model,
    build_roas_heatmap,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TVLift · Convergent TV Attribution",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "tv":          "#7F77DD",
    "facebook":    "#1D9E75",
    "search":      "#D85A30",
    "ooh":         "#BA7517",
    "actual":      "#378ADD",
    "counter":     "#E24B4A",
    "uncertainty": "rgba(55, 138, 221, 0.15)",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── Data loading (all cached) ─────────────────────────────────────────────────

@st.cache_data
def load_data():
    df, adstock_params = load_and_engineer("data/robyn_dt_simulated.csv")
    return df, adstock_params

@st.cache_resource
def load_model(df):
    return train_xgb(df)

@st.cache_data
def load_geo(_df):
    geo_df  = simulate_geo_lift(_df)
    metrics = bootstrap_lift(geo_df, n_iterations=1000)
    return geo_df, metrics

@st.cache_data
def load_power(_geo_df):
    return run_full_power_analysis(_geo_df, target_lift_pct=10.0)

@st.cache_data
def load_daypart(_df):
    airings_df = simulate_airing_level_data(_df)
    model, feature_cols, r2 = train_daypart_model(airings_df)
    heatmap_df = build_roas_heatmap(model, feature_cols, airings_df)
    return airings_df, heatmap_df, r2

@st.cache_data
def load_bayesian(_df, _adstock_params):
    try:
        from tvlift.bayesian_mmm import (
            build_bayesian_mmm,
            extract_channel_roas,
            compute_counterfactual,
        )
        mmm, trace, rev_mean, rev_std, spend_arrays = build_bayesian_mmm(
            _df, _adstock_params,
            sample_kwargs={
                "draws": 500,
                "tune": 300,
                "chains": 2,
                "target_accept": 0.9,
                "return_inferencedata": True,
                "progressbar": False,
            }
        )
        roas_df    = extract_channel_roas(trace, _df, spend_arrays, rev_mean, rev_std)
        counter_df = compute_counterfactual(trace, _df, spend_arrays, rev_mean, rev_std)
        return roas_df, counter_df, True
    except Exception as e:
        st.warning(f"Bayesian sampling error: {e}")
        return None, None, False

# ── Load all data ─────────────────────────────────────────────────────────────
df, adstock_params       = load_data()
model, feature_cols, cv_mape, shap_vals, X_train = load_model(df)
geo_df, geo_metrics      = load_geo(df)
power_df                 = load_power(geo_df)
airings_df, heatmap_df, daypart_r2 = load_daypart(df)
naive                    = naive_attribution(df)
_, coefs, r2, scaler     = ols_attribution(df)
iroas                    = compute_incremental_roas(df, coefs)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## TVLift")
st.sidebar.markdown("*Convergent TV attribution engine*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "🔄 Adstock & saturation",
        "🎯 Attribution",
        "📍 Geo-lift",
        "🔬 Experiment design",
        "🧠 Bayesian MMM",
        "📺 Daypart analysis",
        "💰 Optimizer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** Meta Robyn MMM")
st.sidebar.markdown(f"**Model CV MAPE:** {cv_mape:.1%}")
st.sidebar.markdown(f"**Daypart model R²:** {daypart_r2:.2f}")
st.sidebar.markdown(
    f"**Geo-lift CI:** {geo_metrics['lift_ci_low']}% – {geo_metrics['lift_ci_high']}%"
)

adstock_tv = adstock_params.get("tv_S", {})
if adstock_tv:
    st.sidebar.markdown(f"**TV adstock θ:** {adstock_tv.get('theta', '–')}")
    st.sidebar.markdown(f"**TV Hill α:** {adstock_tv.get('alpha', '–')}")


# PAGE 1 — OVERVIEW
if page == "📊 Overview":
    st.title("TVLift — Convergent TV Attribution Engine")
    st.caption(
        "A production-grade Media Mix Model with geo-lift measurement, "
        "Bayesian uncertainty quantification, and budget optimization. "
        "Built to mirror TV Ad Company's measurement methodology."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total TV spend",   f"${df['tv_S'].sum()/1e6:.1f}M")
    c2.metric("Total revenue",    f"${df['revenue'].sum()/1e6:.0f}M")
    c3.metric(
        "Geo lift (TV)",
        f"{geo_metrics['lift_pct']:.1f}%",
        delta=f"95% CI: {geo_metrics['lift_ci_low']}–{geo_metrics['lift_ci_high']}%",
    )
    c4.metric(
        "Incremental ROAS",
        f"{geo_metrics['incremental_roas']:.2f}x",
        delta=f"CI: {geo_metrics['iroas_ci_low']}–{geo_metrics['iroas_ci_high']}x",
    )
    c5.metric(
        "TV adstock θ",
        f"{adstock_params.get('tv_S', {}).get('theta', '–')}",
        delta="fitted from data",
    )

    # Indexed spend chart
    st.subheader("Weekly spend by channel (indexed to mean = 100)")
    fig = go.Figure()
    for col, label, color in [
        ("tv_S",       "TV",       COLORS["tv"]),
        ("facebook_S", "Facebook", COLORS["facebook"]),
        ("search_S",   "Search",   COLORS["search"]),
        ("ooh_S",      "OOH",      COLORS["ooh"]),
    ]:
        if col in df.columns:
            indexed = df[col] / df[col].mean() * 100
            fig.add_trace(go.Scatter(
                x=df["date"], y=indexed, name=label,
                line=dict(color=color, width=2),
            ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=320,
        yaxis_title="Indexed spend",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")

    # Raw vs adstock-transformed
    if "tv_S_transformed" in df.columns:
        st.subheader("Raw TV spend vs adstock-transformed spend")
        st.caption(
            "The transformed signal captures carryover — "
            "a week with no TV spend still has residual effect from prior weeks."
        )
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(
            x=df["date"], y=df["tv_S"],
            name="Raw TV spend",
            line=dict(color=COLORS["tv"], width=1.5, dash="dot"),
        ))
        fig2.add_trace(go.Scatter(
            x=df["date"], y=df["tv_S_transformed"],
            name="Adstock-transformed",
            line=dict(color=COLORS["facebook"], width=2),
        ), secondary_y=True)
        fig2.update_layout(
            **PLOTLY_LAYOUT, height=300,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig2, width="stretch")


# PAGE 2 — ADSTOCK & SATURATION

elif page == "🔄 Adstock & saturation":
    st.title("Adstock decay & Hill saturation curves")
    st.info(
        "**Why this matters:** TV spend doesn't have an instant effect. "
        "An airing this week still drives conversions next week (adstock). "
        "And spending twice as much doesn't drive twice the response (saturation). "
        "Both effects must be modeled or your ROAS estimates are wrong."
    )

    # Fitted params table
    st.subheader("Fitted adstock parameters")
    params_rows = []
    for col, params in adstock_params.items():
        params_rows.append({
            "Channel":      col.replace("_S", "").upper(),
            "Decay (θ)":    params["theta"],
            "Shape (α)":    params["alpha"],
            "Half-sat (γ)": params["gamma"],
            "Converged":    "✓" if params["converged"] else "✗",
        })
    if params_rows:
        st.dataframe(
            pd.DataFrame(params_rows).set_index("Channel"),
            use_container_width=True,
        )
        st.caption(
            "θ (theta): how fast the TV effect decays. "
            "0 = no carryover, 1 = effect never fades. "
            "α (alpha): shape of diminishing returns curve. "
            "γ (gamma): spend level where you hit 50% of maximum response."
        )

    # Response curves
    st.subheader("Spend → response curves by channel")
    fig = go.Figure()
    colors_list = [COLORS["tv"], COLORS["facebook"], COLORS["search"], COLORS["ooh"]]
    for (col, params), color in zip(adstock_params.items(), colors_list):
        max_spend = df[col].quantile(0.95)
        curve_df  = build_response_curve(params, max_spend)
        fig.add_trace(go.Scatter(
            x=curve_df["spend"], y=curve_df["response"],
            name=col.replace("_S", "").upper(),
            line=dict(color=color, width=2.5),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=360,
        xaxis_title="Weekly spend ($)",
        yaxis_title="Normalized response",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Flat curves = saturation. If TV's curve flattens early, "
        "that spend is wasted at the margin — redirect to other channels."
    )

    # Decay curves
    st.subheader("Adstock decay over time")
    st.caption("How much residual effect remains N weeks after a single airing?")
    weeks = np.arange(0, 13)
    fig2  = go.Figure()
    for (col, params), color in zip(adstock_params.items(), colors_list):
        theta = params["theta"]
        decay = theta ** weeks
        fig2.add_trace(go.Scatter(
            x=weeks, y=decay * 100,
            name=col.replace("_S", "").upper(),
            line=dict(color=color, width=2),
            mode="lines+markers",
        ))
    fig2.update_layout(
        **PLOTLY_LAYOUT, height=300,
        xaxis_title="Weeks after airing",
        yaxis_title="Residual effect (%)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig2, width="stretch")


# PAGE 3 — ATTRIBUTION

elif page == "🎯 Attribution":
    st.title("Attribution — naive vs causal")
    st.info(
        "**The straw man argument:** Naive attribution credits TV proportional "
        "to its spend share. OLS gives correlation, not causation. "
        "The geo-lift experiment on the next page gives true causal credit."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Naive: proportional to spend")
        labels = [k.replace("_S", "").upper() for k in naive.keys()]
        values = list(naive.values())
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.45,
            marker_colors=[
                COLORS["tv"], COLORS["facebook"],
                COLORS["search"], COLORS["ooh"], "#888"
            ],
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Problem: TV gets credit just for being expensive. "
            "A channel could run ads while revenue happened to spike "
            "for unrelated reasons — naive attribution can't distinguish."
        )

    with col2:
        st.subheader("OLS-derived incremental ROAS")
        channels = list(iroas.keys())
        vals     = list(iroas.values())
        fig2 = go.Figure(go.Bar(
            x=[c.replace("_S", "").upper() for c in channels],
            y=vals,
            marker_color=[COLORS["tv"], COLORS["facebook"], COLORS["search"]],
            text=[f"{v:.2f}x" for v in vals],
            textposition="outside",
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT, height=280,
            yaxis_title="Incremental ROAS",
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Better than naive but still correlational. "
            "OLS can't control for omitted variable bias — "
            "TV spend correlates with holiday seasons which also drive revenue."
        )

    # SHAP feature importance
    st.subheader("XGBoost feature importance (SHAP values)")
    st.caption(
        "SHAP decomposes each prediction into individual feature contributions. "
        "Note that adstock-transformed features outrank raw spend — "
        "the model correctly learns that carryover matters."
    )
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_df   = pd.DataFrame({"feature": feature_cols, "importance": mean_shap})
    shap_df   = shap_df.sort_values("importance", ascending=True).tail(12)
    fig3 = go.Figure(go.Bar(
        x=shap_df["importance"], y=shap_df["feature"],
        orientation="h", marker_color=COLORS["tv"],
    ))
    fig3.update_layout(
        **PLOTLY_LAYOUT, height=380,
        xaxis_title="Mean |SHAP value|",
    )
    st.plotly_chart(fig3, width="stretch")


# PAGE 4 — GEO-LIFT

elif page == "📍 Geo-lift":
    st.title("Geo-lift — causal TV measurement")
    st.info(
        "**TV Ad Company's core methodology:** Pause TV in randomly selected holdout DMAs. "
        "Measure the revenue gap vs treatment DMAs. That gap — statistically tested — "
        "is TV's true incremental impact. Bootstrap CI shows how stable the estimate is."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lift estimate", f"{geo_metrics['lift_pct']:.1f}%")
    c2.metric(
        "95% CI",
        f"{geo_metrics['lift_ci_low']}% – {geo_metrics['lift_ci_high']}%",
        delta="bootstrap, 1000 iterations",
    )
    c3.metric(
        "Incremental ROAS",
        f"{geo_metrics['incremental_roas']:.2f}x",
        delta=f"CI: {geo_metrics['iroas_ci_low']}–{geo_metrics['iroas_ci_high']}x",
    )
    c4.metric(
        "p-value",
        f"{geo_metrics['p_value']:.4f}",
        delta="Significant ✓" if geo_metrics["significant"] else "Not significant",
    )

    # Treatment vs holdout over time
    st.subheader("Treatment vs holdout revenue over time")
    weekly = geo_df.groupby(["date", "group"])["observed_revenue"].mean().reset_index()
    fig = go.Figure()
    for group, color in [
        ("treatment", COLORS["actual"]),
        ("holdout",   COLORS["counter"]),
    ]:
        d = weekly[weekly["group"] == group]
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["observed_revenue"],
            name=group.capitalize(),
            line=dict(color=color, width=2.5),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=320,
        yaxis_title="Avg weekly revenue",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")

    # Bootstrap distribution
    st.subheader("Bootstrap distribution of lift estimate")
    st.caption(
        "Each bar = one bootstrap resample of the DMA assignment. "
        "The spread shows how sensitive the estimate is to which DMAs were chosen. "
        "A narrow distribution = robust measurement."
    )
    lift_dist = geo_metrics["lift_distribution"]
    fig2 = go.Figure(go.Histogram(
        x=lift_dist, nbinsx=60,
        marker_color=COLORS["tv"], opacity=0.8,
    ))
    fig2.add_vline(
        x=geo_metrics["lift_pct"],
        line_color=COLORS["actual"], line_dash="solid", line_width=2,
        annotation_text=f"Median: {geo_metrics['lift_pct']}%",
    )
    fig2.add_vline(
        x=geo_metrics["lift_ci_low"],
        line_color="gray", line_dash="dash",
        annotation_text="2.5%",
    )
    fig2.add_vline(
        x=geo_metrics["lift_ci_high"],
        line_color="gray", line_dash="dash",
        annotation_text="97.5%",
    )
    fig2.update_layout(
        **PLOTLY_LAYOUT, height=300,
        xaxis_title="Lift (%)",
        yaxis_title="Bootstrap frequency",
    )
    st.plotly_chart(fig2, width="stretch")

    # What would tighten this estimate
    st.markdown("---")
    st.subheader("What would tighten this estimate?")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Current CI width",
        f"{geo_metrics['lift_ci_high'] - geo_metrics['lift_ci_low']:.1f}pp",
    )
    col2.metric("DMAs needed to halve CI", "~40 total")
    col3.metric("Or extend experiment to",  "8 weeks")
    st.caption(
        "See the Experiment design page to interactively explore the "
        "tradeoff between DMA count, experiment duration, and measurement precision."
    )



# PAGE 5 — EXPERIMENT DESIGN

elif page == "🔬 Experiment design":
    st.title("Pre-experiment power analysis")
    st.info(
        "**Senior DS thinking:** You design the experiment *before* you run it. "
        "Power analysis tells you how many DMAs you need, how long to run, "
        "and what's the smallest lift you can reliably detect. "
        "Skip this and you might run an expensive experiment that proves nothing."
    )

    col1, col2 = st.columns(2)
    with col1:
        target_lift = st.slider(
            "Target lift to detect (%)", 2.0, 30.0, 10.0, 0.5
        )
        alpha_val = st.select_slider(
            "False positive rate (α)", [0.01, 0.05, 0.10], value=0.05
        )
        power_val = st.select_slider(
            "Statistical power (1-β)", [0.70, 0.80, 0.90], value=0.80
        )

    power_curve     = run_full_power_analysis(
        geo_df,
        target_lift_pct=target_lift,
        alpha=alpha_val,
        power=power_val,
    )
    detectable      = power_curve[power_curve["detects_target"]]
    min_dmas_needed = detectable["total_dmas"].min() if len(detectable) else None

    with col2:
        st.metric(
            "Minimum total DMAs needed",
            f"{int(min_dmas_needed)}" if min_dmas_needed else "50+",
            delta=f"to detect {target_lift}% lift at {power_val:.0%} power",
        )

    st.subheader("MDE vs number of DMAs")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=power_curve["total_dmas"],
        y=power_curve["mde_pct"],
        mode="lines+markers",
        line=dict(color=COLORS["tv"], width=2.5),
        name="MDE",
    ))
    fig.add_hline(
        y=target_lift,
        line_color=COLORS["counter"], line_dash="dash",
        annotation_text=f"Target: {target_lift}%",
    )
    if len(detectable):
        fig.add_vrect(
            x0=detectable["total_dmas"].min(),
            x1=power_curve["total_dmas"].max(),
            fillcolor="rgba(29,158,117,0.1)",
            line_width=0,
            annotation_text="Can detect target",
        )
    fig.update_layout(
        **PLOTLY_LAYOUT, height=360,
        xaxis_title="Total DMAs (treatment + holdout)",
        yaxis_title="Minimum detectable effect (%)",
    )
    st.plotly_chart(fig, width="stretch")

    row20 = power_curve[power_curve["total_dmas"] == 20]
    if len(row20):
        mde_at_20 = row20["mde_pct"].values[0]
        st.caption(
            f"Read this chart as: with 20 total DMAs, you can reliably detect "
            f"a lift of {mde_at_20:.1f}% or larger. "
            "If you only expect a 5% lift, you need more DMAs."
        )


# PAGE 6 — BAYESIAN MMM

elif page == "🧠 Bayesian MMM":
    st.title("Bayesian Media Mix Model")
    st.info(
        "**The production standard:** Instead of a point estimate ('TV ROAS = 1.8x'), "
        "Bayesian MMM gives you a full posterior distribution. "
        "You can say 'TV ROAS is 1.8x (94% HDI: 1.1x – 2.7x)' — "
        "which is the honest answer when data is noisy."
    )

    run_bayes = st.button(
        "Run Bayesian MMM sampling (~2 min)",
        help="MCMC sampling. Runs once then cached automatically.",
    )

    if run_bayes or "bayes_ran" in st.session_state:
        st.session_state["bayes_ran"] = True

        with st.spinner("Sampling posterior... (~2 min, cached after first run)"):
            roas_df, counter_df, success = load_bayesian(df, adstock_params)

        if not success or roas_df is None:
            st.error(
                "Bayesian sampling failed. "
                "Check PyMC 5.x is installed: `pip install pymc>=5.0`"
            )
        else:
            # Posterior ROAS bars
            st.subheader("Posterior ROAS by channel")
            st.caption(
                "Each bar = median ROAS. Error bars = 94% highest density interval. "
                "Overlapping intervals = channels are statistically indistinguishable."
            )
            fig = go.Figure()
            channel_colors = {
                "TV": COLORS["tv"], "Facebook": COLORS["facebook"],
                "Search": COLORS["search"], "OOH": COLORS["ooh"],
            }
            for _, row in roas_df.iterrows():
                ch    = row["channel"]
                color = channel_colors.get(ch, "#888")
                fig.add_trace(go.Bar(
                    x=[ch],
                    y=[row["roas_median"]],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[row["roas_hdi_high"] - row["roas_median"]],
                        arrayminus=[row["roas_median"] - row["roas_hdi_low"]],
                    ),
                    marker_color=color,
                    name=ch,
                    text=f"{row['roas_median']:.2f}x",
                    textposition="outside",
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=360,
                yaxis_title="Incremental ROAS",
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

            # Full posterior distributions
            st.subheader("Full posterior distributions")
            fig2 = go.Figure()
            for _, row in roas_df.iterrows():
                ch      = row["channel"]
                color   = channel_colors.get(ch, "#888")
                samples = row["roas_samples"]
                fig2.add_trace(go.Violin(
                    x=[ch] * len(samples),
                    y=samples,
                    name=ch,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=color,
                    opacity=0.7,
                    line_color=color,
                ))
            fig2.update_layout(
                **PLOTLY_LAYOUT, height=360,
                yaxis_title="ROAS samples",
                showlegend=False,
            )
            st.plotly_chart(fig2, width="stretch")

            # Counterfactual chart
            if counter_df is not None:
                st.subheader("Counterfactual: actual vs zero-TV-spend revenue")
                st.caption(
                    "The shaded gap is TV's causal contribution to revenue. "
                    "This is the entire argument for why TV Ad Companies exists."
                )
                fig3 = go.Figure()

                # Uncertainty band
                fig3.add_trace(go.Scatter(
                    x=counter_df["date"], y=counter_df["actual_hdi_high"],
                    mode="lines", line=dict(width=0),
                    showlegend=False, name="Actual upper",
                ))
                fig3.add_trace(go.Scatter(
                    x=counter_df["date"], y=counter_df["actual_hdi_low"],
                    fill="tonexty",
                    fillcolor=COLORS["uncertainty"],
                    line=dict(width=0),
                    showlegend=False, name="Actual lower",
                ))
                # Actual revenue line
                fig3.add_trace(go.Scatter(
                    x=counter_df["date"], y=counter_df["actual_median"],
                    line=dict(color=COLORS["actual"], width=2.5),
                    name="Actual revenue",
                ))
                # Counterfactual line
                fig3.add_trace(go.Scatter(
                    x=counter_df["date"], y=counter_df["counter_median"],
                    line=dict(color=COLORS["counter"], width=2, dash="dash"),
                    name="Without TV",
                ))
                # TV contribution shading
                fig3.add_trace(go.Scatter(
                    x=pd.concat([
                        counter_df["date"],
                        counter_df["date"][::-1],
                    ]),
                    y=pd.concat([
                        counter_df["actual_median"],
                        counter_df["counter_median"][::-1],
                    ]),
                    fill="toself",
                    fillcolor="rgba(127,119,221,0.15)",
                    line=dict(width=0),
                    name="TV contribution",
                ))

                total_tv_contrib = counter_df["tv_contribution"].sum()
                fig3.update_layout(
                    **PLOTLY_LAYOUT, height=420,
                    yaxis_title="Revenue ($)",
                    legend=dict(orientation="h", y=-0.2),
                    annotations=[dict(
                        x=counter_df["date"].median(),
                        y=counter_df["actual_median"].max() * 0.92,
                        text=f"Total TV contribution: ${total_tv_contrib/1e6:.1f}M",
                        showarrow=False,
                        font=dict(color=COLORS["tv"], size=13),
                    )],
                )
                st.plotly_chart(fig3, width="stretch")
    else:
        st.info(
            "Bayesian MCMC sampling takes ~2 minutes and is cached after the first run. "
            "Click the button above to start."
        )

# PAGE 7 — DAYPART ANALYSIS
elif page == "📺 Daypart analysis":
    st.title("Daypart × day-of-week ROAS analysis")
    st.info(
        "**The client question TV Ad Company actually answers:** Not 'does TV work?' "
        "but 'which TV slots work?' Saturday primetime vs Tuesday late night "
        "can differ by 6x in ROAS. This analysis drives actual media buy decisions."
    )

    daypart_order = ["primetime", "daytime", "morning", "late_night"]
    day_order     = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    # ROAS heatmap
    st.subheader("ROAS heatmap by daypart and day of week")
    pivot = heatmap_df.pivot(
        index="daypart", columns="day_of_week", values="predicted_roas"
    ).reindex(index=daypart_order, columns=day_order)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[d.capitalize() for d in day_order],
        y=["Primetime", "Daytime", "Morning", "Late night"],
        colorscale="Purples",
        text=[[f"{v:.2f}x" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(title="ROAS"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Darker = higher ROAS. Saturday primetime is the peak. "
        f"Late night weekday is the floor. Model R²: {daypart_r2:.2f}"
    )

    # Best / worst slots
    st.subheader("Best and worst 5 slots")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 slots**")
        top5 = (
            heatmap_df
            .nlargest(5, "predicted_roas")[["daypart", "day_of_week", "predicted_roas"]]
            .reset_index(drop=True)
        )
        top5.columns = ["Daypart", "Day", "ROAS"]
        st.dataframe(top5, use_container_width=True)

    with col2:
        st.markdown("**Bottom 5 slots**")
        bot5 = (
            heatmap_df
            .nsmallest(5, "predicted_roas")[["daypart", "day_of_week", "predicted_roas"]]
            .reset_index(drop=True)
        )
        bot5.columns = ["Daypart", "Day", "ROAS"]
        st.dataframe(bot5, use_container_width=True)

    # Average ROAS by daypart
    st.subheader("Average ROAS by daypart")
    dp_avg = (
        heatmap_df
        .groupby("daypart")["predicted_roas"]
        .mean()
        .reindex(daypart_order)
        .reset_index()
    )
    fig2 = go.Figure(go.Bar(
        x=["Primetime", "Daytime", "Morning", "Late night"],
        y=dp_avg["predicted_roas"],
        marker_color=[
            COLORS["tv"], COLORS["facebook"],
            COLORS["search"], COLORS["ooh"],
        ],
        text=[f"{v:.2f}x" for v in dp_avg["predicted_roas"]],
        textposition="outside",
    ))
    fig2.update_layout(
        **PLOTLY_LAYOUT, height=320,
        yaxis_title="Predicted ROAS",
    )
    st.plotly_chart(fig2, width="stretch")

    # Key insight callout
    best_slot  = heatmap_df.loc[heatmap_df["predicted_roas"].idxmax()]
    worst_slot = heatmap_df.loc[heatmap_df["predicted_roas"].idxmin()]
    ratio      = best_slot["predicted_roas"] / (worst_slot["predicted_roas"] + 1e-9)
    st.info(
        f"**Key insight:** {best_slot['daypart'].replace('_',' ').title()} "
        f"{best_slot['day_of_week'].capitalize()} is your best slot at "
        f"{best_slot['predicted_roas']:.2f}x ROAS — "
        f"{ratio:.1f}x better than "
        f"{worst_slot['daypart'].replace('_',' ').title()} "
        f"{worst_slot['day_of_week'].capitalize()} "
        f"({worst_slot['predicted_roas']:.2f}x). "
        "Shifting budget toward top slots is the immediate optimization opportunity."
    )



# PAGE 8 — OPTIMIZER
elif page == "💰 Optimizer":
    st.title("Budget optimizer — maximize incremental ROAS")
    st.info(
        "**The punchline:** You can only optimize what you can measure. "
        "Every prior page built up to this — the optimizer uses the "
        "adstock-corrected, saturation-aware model to find the allocation "
        "that maximizes predicted incremental revenue."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        budget = st.slider(
            "Total weekly budget ($)",
            min_value=10_000, max_value=500_000,
            value=100_000, step=5_000,
            format="$%d",
        )
    with col2:
        st.markdown("**Channel bounds**")
        tv_min = st.slider("TV min %", 0, 60, 10)
        tv_max = st.slider("TV max %", 20, 90, 70)

    with st.spinner("Optimizing..."):
        result = optimize_budget(
            model, feature_cols, df,
            total_budget=budget,
            channel_bounds={
                "tv_S":       (tv_min / 100, tv_max / 100),
                "facebook_S": (0.10, 0.50),
                "search_S":   (0.10, 0.50),
            },
        )

    alloc   = result["optimal_allocation"]
    opt_rev = result["optimal_predicted_revenue"]
    eq_rev  = result["equal_split_revenue"]
    uplift  = result["revenue_uplift_pct"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal predicted revenue", f"${opt_rev:,.0f}")
    c2.metric("Equal-split baseline",      f"${eq_rev:,.0f}")
    c3.metric("Revenue uplift",            f"{uplift:.1f}%",
              delta=f"+{uplift:.1f}% from optimization")

    # Allocation bar chart
    st.subheader("Optimal spend allocation")
    labels = ["TV", "Facebook", "Search"]
    values = [alloc["tv_S"], alloc["facebook_S"], alloc["search_S"]]
    colors = [COLORS["tv"], COLORS["facebook"], COLORS["search"]]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"${v:,.0f}\n({v/budget*100:.0f}%)" for v in values],
        textposition="outside",
    ))
    fig.add_hline(
        y=budget / 3, line_dash="dot",
        annotation_text="Equal split baseline",
        line_color="gray",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT, height=340,
        yaxis_title="Spend ($)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Search shows high model ROAS due to its correlation with high-intent "
        "conversion events. In production, TV Ad Company constrains channel allocations "
        "using historical capacity curves and brand-building objectives — "
        "pure ROAS optimization without constraints would over-index performance channels."
    )

    # ROAS landscape heatmap
    st.subheader("ROAS landscape — TV share vs Facebook share")
    st.caption(
        "The darkest cell is the optimal allocation. "
        "Note how ROAS degrades as you move away from the optimum — "
        "this is the cost of sub-optimal TV allocation."
    )
    all_res = result["all_results"]
    if not all_res.empty:
        all_res["tv_pct"] = (all_res["tv_S"] / budget * 100).round(0)
        all_res["fb_pct"] = (all_res["facebook_S"] / budget * 100).round(0)
        pivot = all_res.pivot_table(
            index="fb_pct", columns="tv_pct",
            values="predicted_revenue", aggfunc="max",
        )
        fig2 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale="Purples",
            colorbar=dict(title="Pred. revenue"),
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT, height=380,
            xaxis_title="TV share (%)",
            yaxis_title="Facebook share (%)",
        )
        st.plotly_chart(fig2, width="stretch")