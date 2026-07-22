# WardWatch: Robust Early-Warning Triage Thresholds

A hospital runs an early-warning model on streaming patients. The model emits a risk
score for near-term deterioration, but deployment is not stationary: ED surges, ward
frailty, ICU load, night shifts, and score over-calling change the mix of patients and
the calibration of scores. Your job is to design a deterministic **alert threshold
policy** that decides when the triage team should escalate.

You receive a labeled source-period calibration stream and an unlabeled recent target
stream. The hidden deployment stream is generated from the same target regime, with
labels held back by the evaluator.

## You write a program (stdin -> stdout)

Your program reads ONE JSON public instance from **stdin** and writes ONE JSON policy to
**stdout**. It is run in an isolated sandbox and sees only the public instance.

### Input (stdin) -- public instance

```json
{
  "instance_id": 5030101,
  "scenario": "ed_surge",
  "strata": ["ED|low", "ED|med", "ED|high", "..."],
  "threshold_grid": [0.05, 0.075, "...", 0.8],
  "baseline_threshold": 0.5,
  "score_bins": [0.0, 0.1, "...", 1.0],
  "costs": {
    "fn_cost_by_acuity": {"low": 8.0, "med": 12.0, "high": 17.0},
    "alert_cost": 1.0,
    "tp_credit": 2.0,
    "fatigue_weight": 0.42,
    "fatigue_budget_per_40": 4.2,
    "calibration_weight": 0.34,
    "monotone_weight": 2.0
  },
  "calibration": [
    {"unit": "ED", "acuity": "med", "age_band": "adult",
     "score": 0.173421, "score_bin": 1, "block": 0, "label": 0}
  ],
  "recent_unlabeled": [
    {"unit": "WARD", "acuity": "high", "age_band": "older",
     "score": 0.291882, "score_bin": 2, "block": 1000}
  ]
}
```

Each record has a unit, acuity, age band, model score in `[0,1]`, score decile, and
block id. Calibration records include `label` (`1` means deterioration); recent target
records do not.

### Output (stdout) -- your threshold policy

```json
{
  "default_threshold": 0.22,
  "thresholds": {
    "ED|low": 0.31,
    "ED|med": 0.21,
    "ED|high": 0.15
  }
}
```

`default_threshold` is used for any stratum not listed. `thresholds` may contain any
subset of the public `strata`. A patient alerts iff `score >= threshold(unit|acuity)`.

## Feasibility

A policy is valid iff:

- `default_threshold` is a finite number in `[0.02, 0.98]`;
- `thresholds` is a JSON object whose keys are public stratum names;
- every listed threshold is finite and in `[0.02, 0.98]`.

Wrong shape, extra stratum keys, non-finite values, a crash, timeout, or non-JSON output
scores 0.0 for that instance.

## Objective & scoring (deterministic; maximize)

The evaluator applies your threshold table to a hidden deployment stream. Utility
combines:

- large asymmetric penalties for false negatives, increasing with acuity;
- a cost for each alert and a small true-positive credit;
- a quadratic alert-fatigue penalty when a unit-block exceeds its alert budget;
- score-decile calibration penalties when alert rates by bin do not match the hidden
  event-rate-implied treatment load, plus a monotonicity penalty if higher score bins
  receive lower alert rates.

Per instance, the evaluator computes:

- `u_base`: the fixed conservative `0.50` threshold policy, anchoring Ratio `0.10`;
- `u_ref`: an internal public-data heuristic that fits stratum thresholds on the
  calibration stream with shrinkage and target-score tail adjustment, anchoring Ratio
  `0.80`.

The normalized reward is:

```text
r = clamp(0.10 + 0.70 * (u_candidate - u_base) / (u_ref - u_base), 0, 1)
```

The final `Ratio` is the mean over 12 seeded instances, including harder held-out
target shifts with fewer recent target examples and tighter fatigue budgets. All data
generation and scoring are deterministic CPU code; there is no wall-time or GPU term.

## Strategy ladder

1. **Conservative default** -- threshold `0.50` everywhere; misses many deteriorations
   but controls fatigue, and anchors the baseline.
2. **Global cost-sensitive threshold** -- fit one threshold from labeled calibration
   outcomes; handles asymmetric false-negative cost but ignores stratum shifts.
3. **Stratum table with shrinkage** -- fit thresholds by `unit|acuity`, shrink sparse
   strata toward the global threshold, and use recent unlabeled target scores to adjust
   expected alert load under distribution shift.
4. **Robust calibrated policy** -- combine score-bin calibration, alert-budget
   constraints, monotone smoothing, and shift-aware threshold selection to exceed the
   internal reference without over-alerting.
