# Robust Regression Head: Fit a Loss for Corrupted Sensors

## Story

A calibration lab fits a **linear regression head** that maps a few raw sensor features
`x = (x_0, ..., x_{d-1})` to a target reading:

```
pred(x) = sum_j w[j] * x[j] + b
```

The problem: a fraction of the **training labels are corrupted**. Some sensors log wild,
gross outliers (spikes / dropouts) with no warning flag, so the training targets you receive
contain a mix of good points and grossly wrong ones. The lab's **validation set is clean** —
it comes from the same underlying signal with only the small measurement noise.

Your job is to **design the loss / fitting procedure** for the head so the learned weights
generalize to the clean distribution. A plain squared (L2) loss chases the outliers and
generalizes poorly; a robust loss or an explicit outlier-rejection procedure recovers the true
head. There is no single best loss — the delta / breakdown point / reweighting schedule all
trade off — so many fitting strategies are viable.

## You write a program (isolated)

Your program is a standalone process. It reads **one** JSON object (the public instance) from
**stdin** and writes **one** JSON object (the learned head) to **stdout**. It never sees the
validation set.

### Input (stdin) — the PUBLIC instance
```json
{
  "name": "cal_1001",
  "d": 6,
  "n_train": 60,
  "X_train": [[..d floats..], ...],   // n_train rows of features
  "y_train": [float, ...]             // n_train targets; a fraction are CORRUPTED outliers
}
```

### Output (stdout) — the learned head
```json
{ "w": [float, ...], "b": float }
```
`w` must be a list of exactly `d` finite numbers and `b` a finite number. Prediction on a
validation point `x` is `sum_j w[j]*x[j] + b`.

A submission is **invalid** (scores 0.0 on that instance) if `w` has the wrong length, any
weight or `b` is non-finite (`NaN`/`Inf`), the output is not valid JSON, or the program crashes
or times out.

## Objective — MINIMIZE held-out error

The evaluator holds out a **clean** validation set `(X_test, y_test)` and computes the
mean squared error of your head:

```
obj  = mean_i ( (pred(x_i) - y_test_i)^2 )      # your held-out MSE  (LOWER is better)
```

## Scoring

Per instance, let `base` be the MSE of the trivial "predict the training-label mean" head:

```
base = mean_i ( (mean(y_train) - y_test_i)^2 )
r    = min( 1.0, 0.1 * base / max(obj, 1e-12) )
```

- Predicting the (outlier-skewed) training mean scores about **0.1**.
- A head that recovers the clean signal drives `obj` well below `base` and scores higher.
- The clean validation noise floor is strictly positive, so `obj` can never reach 0 —
  even an excellent robust fit stays **below 1.0** (headroom is left on purpose).

The reported score is the mean of `r` over a fixed, seeded distribution of instances
(easy → heavily corrupted / few-sample held-out cases). Scoring is fully deterministic.

## Notes / strategy hints

- **Ordinary least squares** fits the signal but a squared loss is dominated by the large
  outlier residuals — a weak-to-moderate baseline.
- **Robust losses** (Huber, absolute/LAD, Tukey biweight, quantile) or **explicit rejection**
  (trim high-residual points via a robust scale such as MAD, then refit) are the strong plays.
- Estimating a robust residual **scale** and choosing the reweighting **threshold** are the key
  knobs; there is no universally optimal setting across the instance mix.
