# Forecast Committee: Fusing a Panel of Base Forecasters

## Story
A risk desk runs a **committee of `k` probabilistic forecasters** (base models). For each
upcoming binary event — *did the flagged shipment actually spoil? did the borrower default?* —
every committee member emits a probability forecast `p_ij` in `[0,1]`. Your job is to write the
**fusion rule** that combines the `k` member forecasts into a single committee forecast `q_i` for
every event.

The members are heterogeneous: some are genuine experts whose forecasts track the truth, some are
near-noise, and some are systematically over- or under-confident (miscalibrated) or biased. A naive
equal-weight average lets the weak members drag the committee down. You are given a **labelled
validation history** (past member forecasts together with the realized outcomes) from which to learn
who to trust and how to recalibrate — but you must forecast the future **without ever seeing the
test outcomes**.

There is no single dominant strategy: skill-weighting, logistic/linear stacking, trimming,
rank-fusion and calibration each buy different amounts on different panels, and none reaches the
(unreachable) oracle that is fitted directly on the test outcomes.

## You write a standalone program (stdin -> stdout)
Read ONE JSON object (the public instance) from stdin, write ONE JSON object to stdout.

### Public instance (stdin)
```json
{
  "name": "panel1101",
  "k": 6,                        // number of committee members
  "n_val": 260,                  // labelled validation events
  "n_test": 260,                 // test events to forecast (outcomes HIDDEN)
  "val_pred":  [[p_0, ..., p_{k-1}], ...],   // n_val rows of member forecasts, each in [0,1]
  "val_y":     [y_0, ..., y_{n_val-1}],      // validation OUTCOMES (0/1) -- given to you
  "test_pred": [[p_0, ..., p_{k-1}], ...]    // n_test rows of member forecasts on test events
}
```

### Answer (stdout)
```json
{"forecast": [q_0, q_1, ..., q_{n_test-1}]}
```
`forecast` must be a list of **exactly `n_test`** finite reals, each in `[0,1]` (a proper
probability). Wrong length, a non-finite value (`nan`/`inf`), any value outside `[0,1]`, a crash, a
timeout, or non-JSON output makes that instance score **0.0**.

## Objective — MAXIMIZE
Your fused forecast is judged by the **Brier score** against the hidden test outcomes `y`
(`BS = mean_i (q_i - y_i)^2`, lower is better). Per instance the evaluator forms two references:

- `BS_base` — Brier of the **equal-weight committee mean** (the weak reference), and
- `BS_orac` — residual MSE of a **least-squares linear fuser fitted on the test outcomes**
  (intercept + `k` member columns): a genuinely **unreachable** upper anchor, since it peeks at the
  labels you never see.

The per-instance score is the affine anchor
```
r = clamp( 0.1 + 0.9 * (BS_base - BS_cand) / max(1e-9, BS_base - BS_orac),  0, 1 )
```
so matching the equal-weight mean scores `~0.1`, reaching the oracle scores `1.0` (unreachable), and
doing worse than the mean scores `< 0.1`. The final score is the mean of `r` over all 12 panels
(8 standard + 4 harder held-out panels with more members and thinner validation). **Leave headroom:**
honest fusers stay well below `1.0`.

## Rules
- **Deterministic:** identical stdin must give identical stdout. Do not use wall-clock time, PIDs, or
  unseeded randomness.
- Your program runs **OS-sandboxed** in a fresh subprocess and only ever sees the public instance
  above (validation labels and member forecasts — never the test outcomes). The references and hidden
  outcomes live only in the evaluator process.
- Time limit per instance is generous; keep it simple and finite.

## Scoring harness
`python3 evaluator.py <your_program.py>` prints `Ratio:` (mean score in `[0,1]`) and `Vector:` (the
per-panel scores).
