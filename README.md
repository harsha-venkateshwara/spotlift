# TVLift — Convergent TV Attribution Engine

> A production-grade Media Mix Model that measures the true incremental 
> impact of TV advertising using geo-lift methodology, Bayesian uncertainty 
> quantification, and budget optimization.
> 
> Built to mirror how ad tech companies like Tatari approach convergent 
> TV measurement.

![TVLift Dashboard](assets/overview.png)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tvlift.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

You can't click a TV ad.

When someone watches a commercial on Monday night and visits your website 
on Tuesday, that visit looks identical to organic traffic. Traditional 
attribution just gives TV credit proportional to its spend share — which 
is correlation at best, and completely useless for causal inference.

TVLift solves this with the same methodology production ad tech teams use:
geo-lift experiments, adstock modeling, Bayesian uncertainty quantification,
and saturation-aware budget optimization.

---

## Live Demo

🔗 **[tvlift.streamlit.app](https://tvlift.streamlit.app)**

The app has 8 interactive pages. Each one tells one piece of the same story:
Measure, Model and Optimize

---

## What's Inside

### Page 1 — Overview
Executive summary dashboard. Five KPIs at a glance: total TV spend, 
total revenue, geo-lift confidence interval, incremental ROAS, and 
fitted adstock decay rate. Includes indexed spend chart and raw vs 
adstock-transformed TV spend comparison.

### Page 2 — Adstock & Saturation
Two non-linear TV effects that most attribution models ignore:

- **Geometric adstock**: TV effects decay over weeks, not instantly. 
  An airing this week still drives conversions next week and the week 
  after. The decay rate θ is fitted from data — not assumed.
- **Hill saturation**: Doubling TV spend doesn't double revenue. 
  Diminishing returns are real and modeled explicitly via the Hill 
  function. Both α (shape) and γ (half-saturation point) are fitted 
  from data.

Includes fitted parameter table, spend → response curves per channel, 
and adstock decay visualization over 12 weeks.

### Page 3 — Attribution
Shows exactly why naive attribution fails — then shows the causal 
alternative.

- **Naive**: Proportional to spend. TV gets credit just for being 
  expensive. Completely ignores causation.
- **OLS regression**: Better, but still correlational. TV spend 
  correlates with Q4 holiday seasons — OLS can't separate the two.
- **SHAP feature importance**: XGBoost + SHAP shows that 
  adstock-transformed features outrank raw spend, confirming the 
  model learned that carryover matters.

### Page 4 — Geo-Lift
The causal core of the project. Mirrors Tatari's core measurement 
methodology.

**How it works:**
1. Split DMAs (Designated Market Areas) into treatment and holdout groups
2. Treatment DMAs keep running TV ads as normal
3. Holdout DMAs have TV ads paused for the experiment duration
4. Measure the revenue gap between groups
5. That gap — statistically tested — is TV's true incremental impact

**Bootstrap CI**: 1,000 iterations of DMA resampling to produce a 
95% confidence interval on the lift estimate. A point estimate without 
uncertainty is just a guess.

Includes treatment vs holdout time series, bootstrap distribution 
histogram, and a "what would tighten this estimate" section connecting 
directly to the experiment design page.

### Page 5 — Experiment Design
Pre-experiment power analysis. You design the experiment before you 
run it — not after.

Interactive sliders for:
- **Target lift to detect** (%)
- **False positive rate** (α)
- **Statistical power** (1-β)

Outputs:
- Minimum total DMAs needed
- MDE vs DMA count curve
- Green shading showing the feasible detection region

This page answers: "If I only expect a 10% TV lift, how many holdout 
DMAs do I actually need?"

### Page 6 — Bayesian MMM
Production-grade uncertainty quantification using PyMC.

Instead of a point estimate ("TV ROAS = 1.8x"), the Bayesian model 
gives you a full posterior distribution ("TV ROAS = 1.8x, 94% HDI: 
1.1x – 2.7x").

**Model structure:**
revenue ~ Normal(mu, sigma)
mu = intercept

beta_tv * hill(adstock(tv_spend))
beta_facebook * facebook_norm
beta_search * search_norm
beta_ooh * ooh_norm
seasonality

**Outputs:**
- Posterior ROAS bars with 94% HDI error bars per channel
- Violin plots of full posterior distributions
- Counterfactual revenue chart: actual revenue vs zero-TV-spend 
  scenario — the shaded gap is TV's total causal contribution

> Note: PyMC requires local installation. The deployed app shows 
> illustrative output. Run locally for real posterior sampling.

### Page 7 — Daypart Analysis
Answers the question TV buyers actually ask: not "does TV work?" 
but "which slots work?"

Simulates airing-level data with realistic ground truth ROAS by slot:
- Saturday primetime: 2.8x ROAS
- Tuesday late night: 0.4x ROAS
- 7x spread recovered from airing-level features

**Outputs:**
- ROAS heatmap: 4 dayparts × 7 days = 28 cells
- Top 5 and bottom 5 slots table
- Average ROAS by daypart bar chart
- Key insight callout with best/worst slot comparison

### Page 8 — Budget Optimizer
The payoff of everything before it. You can only optimize what 
you can measure.

Given a total weekly budget and configurable channel bounds, 
find the spend allocation across TV, Facebook, and Search that 
maximizes predicted incremental revenue.

**Features:**
- Budget slider: $10K – $500K
- Per-channel min/max bounds (range sliders for all 3 channels)
- Feasibility validation — warns if bounds are infeasible
- Allocation bar chart with equal-split baseline comparison
- Summary table showing which channels are at their bounds
- ROAS landscape heatmap (TV share vs Facebook share)

The optimizer uses adstock-corrected, saturation-aware features — 
it knows the 1,000th dollar of TV spend is worth less than the first.

## Technical Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| Visualization | Plotly |
| Causal model | Geo-lift + bootstrap CI |
| Bayesian MMM | PyMC 5.x |
| Predictive model | XGBoost |
| Explainability | SHAP |
| Adstock fitting | scipy.optimize L-BFGS-B |
| Power analysis | scipy.stats |
| Data | Meta Robyn MMM (open source) |
| Language | Python 3.11 |

---

## Methodology

### Why geo-lift over MMM alone?

MMM (Media Mix Modeling) gives you correlation. Geo-lift gives you 
causation. The gold standard is both — use geo-lift to validate MMM 
coefficients, and MMM to generalize geo-lift findings across time.

TVLift implements both:
- Geo-lift for causal identification
- Bayesian MMM for uncertainty quantification and generalization
- XGBoost for prediction and optimization

### Why Bayesian over frequentist MMM?

Frequentist MMM gives you a point estimate and a p-value. Bayesian MMM 
gives you a full posterior distribution — which means you can say 
"there's a 94% probability TV ROAS is between 1.1x and 2.7x" rather 
than "TV ROAS is 1.8x, p < 0.05."

For budget decisions involving millions of dollars, the honest 
uncertainty range matters more than the point estimate.

### Why adstock and saturation matter

**Without adstock**: The model assumes a TV airing this week has 
zero effect next week. This underestimates TV's true contribution 
and produces artificially low ROAS estimates.

**Without saturation**: The optimizer assumes linear returns and 
pours the entire budget into whichever channel has the highest 
marginal ROAS — producing unrealistic recommendations.

Both effects are fitted from data using scipy L-BFGS-B optimization, 
not assumed from industry benchmarks.

---

## Running Locally

### Prerequisites
- Python 3.11
- Git

### Installation
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/tvlift.git
cd tvlift

# Install dependencies
pip install -r requirements.txt

# For Bayesian MMM (optional, adds ~2 min sampling)
pip install pymc==5.10.0 pytensor==2.18.0
```

### Download the dataset

The dataset downloads automatically on first run. Or manually:
```bash
mkdir data
curl -o data/robyn_dt_simulated.csv \
  "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/robyn_package/data-raw/dt_simulated_weekly.csv"
```

### Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Dataset

**Meta Robyn MMM Dataset** — open source, released by Meta's 
Marketing Science team as part of the 
[Robyn](https://github.com/facebookexperimental/Robyn) project.

Contains simulated weekly data for a DTC brand:
- TV, Facebook, Search, OOH, Print spend
- Weekly revenue
- Holiday and event flags
- 4+ years of weekly observations

This dataset is the industry standard for MMM research and is used 
in academic papers, open-source libraries, and production model 
validation.

---

## Key Results

| Metric | Value |
|---|---|
| TV geo-lift | 56.3% (95% CI: 32.9% – 82.1%) |
| TV adstock decay θ | 0.76 (high carryover) |
| TV Hill shape α | 0.68 (concave returns) |
| XGBoost CV MAPE | 19.0% |
| Daypart model R² | 0.97 |
| Best slot | Saturday primetime (2.8x ROAS) |
| Worst slot | Late night weekday (0.4x ROAS) |
| Budget optimization uplift | 3–8% vs equal split |


## What I'd Add to this in future

- **ACR data integration**: Real household-level TV viewing data 
  from companies like Samba TV or iSpot would replace the simulated 
  airing data with actual impressions
- **Hierarchical Bayesian model**: Share information across DMAs 
  and time periods using a multilevel model
- **Frequency capping analysis**: Model diminishing returns as a 
  function of ad frequency, not just spend
- **CTV vs linear decomposition**: Separate streaming TV from 
  linear TV within the model — Tatari's core convergent TV insight
- **Real-time optimization**: Move from weekly batch optimization 
  to within-week reallocation as response data comes in

---

## About

Built by **Harsha Venkateshwara** as a portfolio project.

MS Computer Science @ University at Buffalo (graduating December 2026)
Specialization: Artificial Intelligence and Machine Learning

---

## License

MIT License — use freely, attribution appreciated.

---

## Acknowledgments

- **Meta Robyn Team** for the open-source MMM dataset and methodology
- **PyMC Team** for the Bayesian modeling framework
- **Tatari** for pioneering convergent TV measurement and inspiring 
  this project's methodology

---

*"You can only optimize what you can measure."*
