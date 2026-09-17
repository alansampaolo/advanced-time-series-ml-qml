# Maximum Likelihood vs Quasi-Maximum Likelihood in Time Series

Simulation based study comparing Maximum Likelihood (ML) and Quasi-Maximum Likelihood (QML) estimation in an AR(1) process with time varying volatility.

## Project Overview

This project investigates how Maximum Likelihood and Quasi-Maximum Likelihood estimators behave when a time-series model is correctly specified versus deliberately misspecified.

The analysis is conducted in a simulation framework, allowing the true data-generating process and parameter values to be known.

The main objective is to compare ML and QML in terms of:

- parameter recovery;
- finite sample efficiency;
- statistical inference;
- robustness to model misspecification;
- asymptotic behavior.

## Data-Generating Process

The simulated process follows an AR(1) model with time-varying volatility:

\[
y_t = c + \phi y_{t-1} + \varepsilon_t
\]

with

\[
\varepsilon_t = \sigma_t z_t, \qquad z_t \sim N(0,1)
\]

and volatility dynamics:

\[
\sigma_t =
\sqrt{\kappa + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2}.
\]

The baseline parameters are:

- \(c = 4\)
- \(\phi = 0.9\)
- \(\kappa = 0.75\)
- \(\alpha = 0.3\)
- \(\beta = 0.6\)

A total of 800 observations are simulated and the first 50 are discarded as burn-in, leaving a sample of 750 observations.

## Maximum Likelihood

Under correct specification, the conditional log-likelihood explicitly incorporates the time varying volatility process.

The parameters

\[
(c,\phi,\kappa,\alpha,\beta)
\]

are estimated numerically by Maximum Likelihood.

Inference is based on the Hessian-based covariance matrix.

The exercise is performed for:

- a finite sample of \(T=750\);
- a large sample of \(T=49,950\).

The large-sample experiment illustrates convergence of the estimates toward the true parameter values and decreasing standard errors.

## Quasi-Maximum Likelihood

The same simulated data are then intentionally estimated using a misspecified homoskedastic AR(1) model:

\[
y_t = c + \phi y_{t-1} + \varepsilon_t,
\qquad
\varepsilon_t \sim N(0,\sigma^2).
\]

The volatility dynamics are therefore ignored.

Although the conditional variance is misspecified, the conditional mean remains correctly specified:

\[
E[y_t \mid y_{t-1}] = c + \phi y_{t-1}.
\]

For this reason, QML can still consistently recover the mean parameters \(c\) and \(\phi\).

Because the likelihood is misspecified, inference is performed using a robust sandwich covariance matrix rather than the standard Hessian-based ML covariance matrix.

## Hypothesis Testing

For both ML and QML, the project tests:

\[
H_0: \phi = 0.8
\]

at the 5% significance level.

In the finite sample:

- ML rejects the null hypothesis;
- QML does not reject the null hypothesis.

This difference reflects the larger standard errors produced under model misspecification and the corresponding loss of estimation efficiency.

## Monte Carlo Experiment

A Monte Carlo experiment with 500 independently simulated samples is used to compare the sampling distributions of the ML and QML estimators.

For every simulated dataset:

1. the correctly specified model is estimated by ML;
2. the misspecified homoskedastic AR(1) model is estimated by QML;
3. estimates of \(c\) and \(\phi\) are stored;
4. kernel density estimates are used to compare their sampling distributions.

The ML distributions are more concentrated around the true parameter values, while the QML distributions are more dispersed in finite samples.

The differences become substantially smaller as sample size increases.

## Main Findings

- Correctly specified ML provides more efficient parameter estimates.
- Model misspecification increases standard errors and reduces finite sample efficiency.
- QML remains consistent for parameters belonging to the correctly specified conditional mean.
- Robust covariance estimation is necessary for valid QML inference under misspecification.
- The finite-sample gap between ML and QML becomes much smaller as the sample size increases.

## Repository Structure

```text
advanced-time-series-ml-qml/
│
├── README.md
├── main.m
├── simulateSV.m
├── LLcontributions.m
├── LLtotal.m
├── LLtotal_NEG.m
├── LL_QMLcontributions.m
├── LL_QMLtotal_NEG.m
└── advanced_time_series_report.pdf
```

### Main files

- `main.m` — complete workflow including simulation, ML estimation, QML estimation, inference and Monte Carlo analysis
- `simulateSV.m` — simulation of the AR(1) process with time varying volatility
- `LLcontributions.m` — conditional ML log-likelihood contributions
- `LLtotal.m` — total conditional ML log-likelihood
- `LLtotal_NEG.m` — negative ML log-likelihood used for numerical minimization
- `LL_QMLcontributions.m` — QML contributions under the misspecified homoskedastic model
- `LL_QMLtotal_NEG.m` — negative QML objective
- `advanced_time_series_report.pdf` — full project report and interpretation

## Dependencies

- MATLAB
- Statistics and Machine Learning Toolbox
- course-provided CML estimation routines

The external CML routines used for numerical estimation are not included in this repository.

## Limitations

This is a simulation study rather than an empirical application to observed financial data.

The comparison is therefore designed to study estimator behavior under controlled conditions rather than to estimate a real world economic relationship.
