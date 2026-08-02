# Passive-Film Breakdown — how long the pipe lasts in a chemistry never tested

## Problem

A steel coupon is protected by a thin passive oxide film. While the film
holds, its corrosion rate `R` (mm/yr) is a smooth, low function of chloride
concentration `Cl`, temperature `T` (°C), `pH`, and exposure time `tex`
(days). But every passive film has a chloride threshold `Cl_crit(T, pH)` —
itself unknown and dependent on `T` and `pH` — past which the film breaks
down locally (pitting) and `R` jumps by **orders of magnitude** above the
passive-regime trend.

Each row of your log also reports `D`, an electrochemical **repassivation
margin** measured on that coupon: `D = (Cl_crit - Cl) / Cl_crit` for the
threshold of that (unknown) chemistry. `D > 0` means the film is intact with
that fractional margin still in hand. Your logbook was pulled entirely from
coupons that never broke down, so `D > 0` on every row you see — and because
the excess-over-threshold is exactly `0` whenever `D >= 0`, the rate `R` in
your log does **not** actually depend on `D` at all. It only starts to matter
once `D` goes negative — in a chemistry your log never tested.

Your job: given the log, output a closed-form expression predicting `R` from
`Cl, T, pH, tex, D` that stays accurate in a **held-out, hotter/more extreme
chemistry** you never saw, where some environments cross the threshold.

## Input (stdin)

```
n_train test_id
Cl_1 T_1 pH_1 tex_1 D_1 R_1
...
```

`n_train` is 28–46 rows. In the TRAINING rows you see: `Cl` roughly in
`[5, 500]`, `T` in `[10, 70]` °C, `pH` in `[5, 9]`, `tex` in `[5, 500]` days,
`D` in `(0, 1]` (small measurement noise keeps it positive), `R > 0`.

The GRADING environments (never shown to you) extrapolate beyond this: `T`
up to `~95` °C, `pH` down to `~3.5` or up to `~10.5`, `Cl` up to roughly
`650`, and `D` as low as about `-1` once the threshold is crossed. Your
expression must stay finite and positive over this wider domain, not just
over the training ranges above.

## Output (stdout): ONE closed-form expression

Print a single line: a Python-style arithmetic expression over the variables
`Cl T pH tex D`, operators `+ - * / **`, unary minus, parentheses, and the
functions `exp(.) log(.) sqrt(.) abs(.) max(.,.) min(.,.)`. Separate tokens
with spaces. No other names, imports, or statements.

**Illustrative FORM only — NOT the hidden law:**
```
2.0 + 0.5 * sin ( Cl )
```
(`sin` isn't even in the allowed function set — this just shows arithmetic
syntax; the real law's shape is different and must be discovered from data
plus the mechanism note above.)

## Feasibility

The expression must parse under the grammar above and evaluate to a finite,
strictly positive value on **every** graded environment (a corrosion rate
cannot be non-finite or negative). Any parse failure, disallowed name/call,
or non-finite/non-positive prediction anywhere in the held-out set scores
`Ratio: 0.0`.

## Objective (maximize)

The grader evaluates your expression on a held-out set of environments the
log never showed you: some near the threshold, some clearly past it at the
SAME `T, pH` range you trained on, and some in an entirely untested,
hotter/more-extreme-pH range that is always past its threshold. Let
`e_i = |log10(pred_i) - log10(R_i)|` over these `M` points, and
`MAE = mean(e_i)`. The grader also computes `MAE_baseline` for its own
constant predictor (the geometric mean of the training `R` you were given —
a "nothing changes" guess). Then:

```
Ratio = min(850, 100 * MAE_baseline / max(1e-9, MAE)) / 1000
```

A constant/no-signal predictor reproduces the baseline (`Ratio ≈ 0.1`).
Lower held-out log-error raises the score; the 850 soft cap (`≤ 0.85`) leaves
headroom, since the exact jump-rate and curvature of the breakdown are never
revealed by an all-passive log.

## Example (worked score, illustrative numbers only)

Suppose `M = 2`, true rates `R = [1.0, 100.0]`, your predictions
`pred = [1.0, 10.0]`. Then `e = [0, 1]`, `MAE = 0.5`. If the constant
baseline (say `5.0`) gives `e = [0.70, 1.30]`, `MAE_baseline = 1.0`. Then
`Ratio = min(850, 100 * 1.0 / 0.5) / 1000 = min(850, 200) / 1000 = 0.2`.

## Constraints

Time limit 4 s, memory 512 MB. `n_train ≤ 46`; expression ≤ 20000 bytes and
≤ 150 AST nodes. Scoring is fully deterministic (all randomness seeded by
`test_id`).
