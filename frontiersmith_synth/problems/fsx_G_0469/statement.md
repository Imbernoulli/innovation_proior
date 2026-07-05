# Recalibrating a Weather Model's Rain Odds

A numerical weather prediction (NWP) system issues, for each forecast window, a raw
**probability of precipitation** (PoP) `s in (0,1)` — its own estimate that it will
rain. Raw NWP probabilities are notoriously **miscalibrated**: depending on season,
region and internal ensemble spread they are systematically **over-confident**
(pushed toward 0/1), **under-confident** (hugging the base rate), or **biased** (a
persistent wet or dry offset).

Your job: write a program that designs a **post-hoc calibration map** and applies it
to a batch of raw test PoPs, using a labelled validation history — WITHOUT ever
seeing the test outcomes. You are graded by a proper score (Brier) on the realized
test outcomes.

## Program contract (stdin → stdout, isolated)

Read ONE JSON object from **stdin** (the public instance):

```json
{"name": "overconf_A",
 "n_val": 300,
 "n_test": 300,
 "val_score":  [0.71, 0.12, ...],   // raw model PoP in (0,1), length n_val
 "val_y":      [1, 0, ...],         // validation outcomes (1 = it rained), length n_val
 "test_score": [0.44, 0.88, ...]}   // raw model PoP on the test windows, length n_test
```

Write ONE JSON object to **stdout**:

```json
{"prob": [q_0, q_1, ..., q_{n_test-1}]}
```

`prob[i]` is your corrected probability of precipitation for test window `i`. Each
value must be a **finite real in [0, 1]**, and the list length must equal `n_test`.

Any of: wrong length, a non-finite value (`nan`/`inf`), a value outside `[0,1]`, a
crash, a timeout, or non-JSON output ⇒ that instance scores **0.0**.

Your program only ever receives the **public** instance (validation scores + labels
and the raw test scores). The test outcomes stay hidden in the evaluator.

## Objective (minimize) and scoring

For each instance, with hidden test outcomes `y`:

- `BS(q) = mean_i (q_i - y_i)^2` — the Brier score, lower is better.
- `BS_base` = Brier of the **raw** model PoP (the identity map) — the weak reference.
- `BS_orac` = Brier of the best **monotone-in-score** map fitted directly on the
  **test** outcomes (isotonic / pool-adjacent-violators regression) — an
  unreachable ideal, since it uses the held-out labels you never see.
- `BS_cand` = Brier of your corrected probabilities.

Your per-instance score is the affine anchor

```
r = clamp( 0.1 + 0.9 * (BS_base - BS_cand) / max(1e-9, BS_base - BS_orac), 0, 1 )
```

so leaving the raw scores untouched scores ~0.1, matching the test-fitted monotone
oracle would score 1.0, and doing worse than the raw model scores below 0.1. The
final Ratio is the mean of `r` over all instances.

Brier is the proper scoring rule whose reliability term is the calibration error but
which — unlike a bare Expected-Calibration-Error — also rewards **sharpness**, so the
degenerate "always predict the base rate" map is bad, not optimal. You must fix the
model's over/under-confidence and bias **without discarding the ranking information**
in the raw score, and generalize from a finite, possibly shifted validation history.

## Notes

- Instances span over-confident, under-confident, wet-biased and dry-biased regimes,
  some with a mild validation→test distribution shift and thin validation histories.
- Scoring is fully deterministic. Seed any randomness you use.
- Multiple strategies are viable — Platt / temperature scaling, histogram binning,
  isotonic regression, beta calibration, shrinkage-to-prior — and buy different
  amounts on different regimes. None reaches the held-out oracle.
