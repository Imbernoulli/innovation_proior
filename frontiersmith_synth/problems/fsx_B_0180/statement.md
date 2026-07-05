# Geothermal Thermal-Decline Scaling Law: Extrapolation

## Problem
Operators of a geothermal field measure the **specific thermal decline** `y`
(degrees C lost per MWh produced) as they push the produced-fluid throughput
scale `x` (a dimensionless load, roughly kg/s) higher. Empirically, `y` falls off
as the throughput grows but flattens toward a **nonzero irreducible floor** caused
by conductive recharge from the surrounding rock.

You are given a table of `(x, y)` measurements from the **small-to-mid throughput
regime** of one field. Propose and fit a single **closed-form scaling law**
`y = f(x)` that best predicts the decline in the **large-throughput regime**,
which is held out from you. You are scored purely on how well your law
**extrapolates** to that unseen region — memorizing the training points does not
help.

## Input (stdin)
```
field_id  n_train
x_1  y_1
x_2  y_2
 ...
x_{n_train}  y_{n_train}
```
`field_id` is an opaque dataset label (an integer; you may use it only as an id —
it carries no physical meaning). `n_train` rows of positive real `x` and `y`
follow, sorted by increasing `x`. Typical instances have `n_train = 24` with `x`
in roughly `[10, 250]`.

## Output (stdout)
A single line: a **closed-form expression in the variable `x`** giving your
predicted `y`. Allowed tokens:
- the variable `x`;
- numeric literals;
- operators `+  -  *  /  **` and parentheses;
- functions `exp`, `log` (natural), `sqrt`; constants `e`, `pi`.

Example output line:
```
2.3 + 58.0 * x ** -0.53
```

No other names are permitted. The expression must evaluate to a finite real
number for every held-out `x` (all held-out `x` are positive and larger than the
training range).

## Feasibility
The submission is rejected (score `0`) if: it is empty; fails to parse; exceeds
4000 characters; contains a disallowed name/token; or evaluates to a non-finite
or out-of-range (`|y| > 1e12`) value at any held-out point.

## Objective
Let `S` be the deterministic held-out (extrapolation) set of throughput values,
larger than every training `x`, with true declines `y*` regenerated from the
field's hidden law (plus fixed, irreducible measurement scatter). Define
```
F_raw = sqrt( mean_{x in S} ( f(x) - y*(x) )^2 )      # held-out RMSE
F     = F_raw * (1 + 0.02 * (#operators in your expression))
```
The `#operators` term is a mild complexity penalty (counts `+ - * /` characters
and the function names `exp`/`log`/`sqrt`); it discourages over-parameterized
memorization. **Smaller `F` is better.**

## Scoring
Let `B` be the held-out RMSE of the internal **flat-persistence baseline** (a
constant equal to the mean of the last three training `y` values). The score is
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000        # printed on the final line
```
So the flat-persistence baseline scores `Ratio ≈ 0.1`; a law with one tenth of
the baseline's held-out error caps at `Ratio = 1.0`. Irreducible measurement
scatter keeps the best achievable score strictly below the cap.

## Constraints
- Deterministic scoring; no timing or randomness.
- `1 ≤ field_id ≤ 8`, `n_train = 24`.
- Held-out set: 24 points with `x` in `[400, 3000]` (log-spaced), disjoint from
  and larger than the training range.

## Example (worked score, illustrative FORM only — NOT the hidden law)
Suppose (for illustration) a solver submits the log+linear shape
`12.0 - 0.8 * log(x) + 0.05 * x` and the checker finds `B = 2.0`, `F_raw = 1.6`
with `4` operators, so `F = 1.6 * (1 + 0.02*4) = 1.728`. Then
`sc = 100 * 2.0 / 1.728 = 115.74`, giving `Ratio = 0.1157`. This expression shape
is shown only to illustrate the input/output and scoring mechanics; the true
scaling law has a different functional form that you must discover from the data.
