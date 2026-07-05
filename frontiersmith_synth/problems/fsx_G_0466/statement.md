# Grid Load Forecasting: Minimum-MASE Hourly Energy-Demand Forecaster

You operate the short-term load forecaster for a synthetic power grid. Every
instance gives you the recent **hourly electricity demand** history of one grid
and asks you to forecast the demand for the next `horizon` hours. The synthetic
load contains the usual structure of real energy demand: a slow growth/decline
**trend**, a double-peak **daily profile** (morning and evening peaks, period =
24 h), a **weekday/weekend calendar effect**, a slow **temperature-driven swing**,
and observation **noise**.

Write a standalone program that reads one instance and prints one forecast.

## Candidate contract (isolated program)

Read ONE JSON object from **stdin**, write ONE JSON object to **stdout**.
The program is run in an isolated sandbox and sees only the public fields below.

### Public instance (stdin)
```json
{
  "y":       [<float>, ...],   // observed hourly demand, contiguous, oldest first
  "period":  24,               // daily season length (hours)
  "horizon": 48                // number of future hours to forecast
}
```
The history covers integer hour indices `0 .. n-1` where `n = len(y)`. Your
forecast must cover the immediately following hours `n .. n+horizon-1`. The
day-of-week of any hour `t` is `(t // 24) % 7` (indices `5,6` are the weekend),
so the calendar phase is recoverable from position alone.

### Answer (stdout)
```json
{ "forecast": [<float>, ...] }   // exactly `horizon` finite numbers
```
Any wrong length, non-numeric, or non-finite (`NaN`/`Inf`) entry scores 0 on
that instance.

## Objective — MINIMIZE MASE

The evaluator holds out the true future demand `a[0..horizon-1]` and scores your
forecast `f` with the **Mean Absolute Scaled Error**:

```
MASE = ( mean_i |f[i] - a[i]| ) / q ,   q = mean_{t=period..n-1} |y[t] - y[t-period]|
```

where `q` is the in-sample seasonal-naive MAE (the natural scale of the series).
**Lower MASE is better.**

## Scoring

Each instance is normalized against the **seasonal-naive** baseline `b` (repeat
the last observed day):

```
score_i = min(1.0, 0.1 * b_i / MASE_i)
```

so simply repeating the last day scores ~0.1, and cutting the error below the
seasonal-naive level (by modelling trend, calendar and slow drift) scores
higher. The reported `Ratio` is the mean of `score_i` over 10 seeded instances
(a mix of clean, structured series and noisy held-out ones). Scoring is fully
deterministic.

## Ideas
- **Seasonal-naive** (baseline): `f[i] = y[n - period + (i % period)]`.
- **Smoothed profile + drift**: average each hour-of-day over the last few days
  and add a linear trend estimated from daily means.
- **Harmonic + calendar regression**: least-squares fit of trend, daily
  harmonics, day-of-week dummies and slow low-frequency terms, then extrapolate.
- **Damped-trend / state-space decomposition** of trend, multi-seasonality and
  the slow temperature component.
