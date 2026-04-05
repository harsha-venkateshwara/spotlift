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

---

## Architecture

tvlift/
├── adstock.py          # Geometric adstock + Hill saturation
│                       # fit_adstock_params, build_response_curve
├── attribution.py      # Naive, OLS, incremental ROAS
├── bayesian_mmm.py     # PyMC model, posterior extraction,
│                       # counterfactual computation
├── daypart.py          # Airing simulation, GBM model,
│                       # ROAS heatmap builder
├── geo_lift.py         # DMA simulation, bootstrap CI
├── model.py            # XGBoost with TimeSeriesSplit CV + SHAP
├── optimizer.py        # Grid search budget optimizer
├── pipeline.py         # Data loading, adstock feature engineering
└── power_analysis.py   # MDE, required DMAs, power curves
data/
└── robyn_dt_simulated.csv   # Meta Robyn MMM open-source dataset
app.py                  # Streamlit dashboard (8 pages)
requirements.txt        # Dependencies

---

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

---

## Interview Questions This Project Answers

If you're using this project for interviews, here are the questions 
you'll get and the answers:

**Q: How do you estimate baseline without a control group?**
A: The geo-lift design creates the control group — holdout DMAs 
serve as the counterfactual. The bootstrap CI quantifies how 
sensitive the estimate is to the specific DMA assignment.

**Q: Why is attribution hard for TV specifically?**
A: Three reasons. First, no click-through tracking — TV impressions 
can't be tied to individual user actions. Second, adstock — effects 
persist across multiple weeks making causal attribution temporally 
ambiguous. Third, confounding — TV spend is correlated with 
seasonality, making OLS estimates unreliable.

**Q: What assumptions does your geo-lift model make?**
A: SUTVA (Stable Unit Treatment Value Assumption) — that DMAs don't 
influence each other. Parallel trends — that treatment and holdout 
DMAs would have followed similar revenue trajectories without the 
experiment. Random DMA assignment — that holdout DMAs aren't 
systematically different from treatment DMAs.

**Q: How would you improve this with real data?**
A: With ACR (Automatic Content Recognition) data — actual household-level 
TV viewing records — you could move from DMA-level to household-level 
attribution. You could also incorporate search query volume as a 
leading indicator of TV response, and use hierarchical Bayesian models 
to share information across DMAs.

**Q: Why grid search instead of convex optimization?**
A: The response surface is non-convex due to the Hill saturation 
transformation — standard convex solvers don't guarantee a global 
optimum. Grid search over allocation fractions is slower but 
guaranteed to find the global optimum within the grid resolution. 
In production you'd use Bayesian optimization for higher-dimensional 
budget problems.

---

## What I'd Add With More Time

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

Built by **Harsha Venkateshwara** as a portfolio project

MS Computer Science @ University at Buffalo (graduating December 2026)
Specialization: Artificial Intelligence and Machine Learning



## Acknowledgments

- **Meta Robyn Team** for the open-source MMM dataset and methodology
- **PyMC Team** for the Bayesian modeling framework
- **Tatari** for pioneering convergent TV measurement and inspiring 
  this project's methodology

---

*"You can only optimize what you can measure."*