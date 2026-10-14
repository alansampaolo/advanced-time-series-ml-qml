# ML vs QML Estimation under Variance Misspecification

Course project, Advanced Time Series Analysis — University of Tübingen (2025) · MATLAB

## Objective

The aim was to **study how Maximum Likelihood (ML) behaves under correct specification compared with Quasi-Maximum Likelihood (QML) under misspecification, and how this affects estimator accuracy**.

## Setup

| | |
|---|---|
| **Data-generating process** | AR(1) with GARCH-type volatility: `y_t = c + φ·y_(t−1) + ε_t`, `ε_t = σ_t·z_t`, `σ_t² = κ + α·ε_(t−1)² + β·σ_(t−1)²` (c = 4, φ = 0.9, κ = 0.75, α = 0.3, β = 0.6) |
| **ML** | Correctly specified model, Hessian-based standard errors |
| **QML** | Homoskedastic AR(1) that ignores the volatility dynamics (the conditional mean is still correct), robust sandwich standard errors |
| **Samples** | T = 750 and T = 49,950, plus a Monte Carlo experiment with 500 replications at T = 750 |

## Key results

- **Both methods recover the mean parameters:** at T = 750, φ̂ = 0.867 (ML) and 0.872 (QML) vs. a true value of 0.9, because the conditional mean is correctly specified.
- **Misspecification costs efficiency:** the QML standard error of φ̂ is about three times larger (0.050 vs. 0.017).
- **This changes inference:** H0: φ = 0.8 is rejected by ML (t ≈ 3.86) but not by QML (t ≈ 1.45), i.e. a higher Type II error.
- **The gap closes with sample size:** at T = 49,950 both estimates of φ are ≈ 0.90, with standard errors of 0.002–0.003.
- **Monte Carlo:** ML sampling distributions are tightly concentrated around the true values, while QML ones are more dispersed and slightly shifted.

## Repository

- `main.m` — full workflow (simulation, ML, QML, inference, Monte Carlo)
- `simulateSV.m` — simulation of the AR(1) process
- `LLcontributions.m`, `LLtotal.m`, `LLtotal_NEG.m` — ML log-likelihood
- `LL_QMLcontributions.m`, `LL_QMLtotal_NEG.m` — QML log-likelihood
- `advanced_time_series_report.pdf` — full report

The optimisation and covariance routines (CML toolbox) were provided by the course and are not included.

**Tools:** MATLAB · Statistics and Machine Learning Toolbox
