# Call-Center Hold Times Near Saturation — recovering a censored divergence

A call center runs at a load level `rho` ("fraction of staffed capacity
used"). There is a load `rho_max` — never published, never measured directly
— above which the center cannot keep up and the *mean* hold time blows up.
Below that ceiling, the mean hold time follows a near-saturation law:

```
mean_hold(rho) = C / (1 - rho / rho_max)
```

for hidden positive constants `C` and `rho_max`. Any *individual* caller's
hold time is a noisy draw around that mean (hold times are high-variance,
not a tight line around a curve). On top of that, the center enforces a hard
patience cap `T`: any caller who would have waited longer than `T` hangs up,
and the logger has no way to record how long they *would* have waited — it
just writes down `T`. This is right-censoring: values above the cap are not
missing, they are systematically replaced by the cap itself.

You are given many `(rho, hold)` observations, logged while the center ran
at **comfortable** loads (well below `rho_max`). Predict `mean_hold(rho)`
well enough to survive grading at **near-critical** loads — closer to the
true (still-hidden) `rho_max` than anything in your training data.

## Input (stdin)

```
n_train  test_id  T
rho_1  hold_1
rho_2  hold_2
...
rho_n  hold_n
```

All values are floats; `rho_i >= 0`, `hold_i >= 0`. `T` is the publicly
known hang-up cap. A row with `hold_i` exactly equal to `T` (to printed
precision) means that caller hung up — the true wait would have been `T` or
more. The held-out grading loads are generated only by the checker and are
never given to you.

## Output (stdout): one expression

Print a single line containing a Python-style expression over the variable
`rho`. Allowed: `+ - * / **`, parentheses, numeric constants, the
one-argument functions `log`, `sqrt`, `exp`, `abs`, comparisons
(`<`, `<=`, `>`, `>=`, `==`, `!=`, no chaining), and a ternary
`A if cond else B`. No other names, calls, statements, or assignments.

**Illustrative FORM only — NOT the hidden law:**

```
0.4 + 0.1 * rho + 0.02 * rho ** 2
```

This only shows the syntax (a bounded polynomial). The real mechanism has an
unbounded divergence at an unknown location and you must discover its shape
and pole from the data.

## Feasibility

Your expression must parse under the grammar above using only known names.
Every value it produces on the held-out set must be finite. Any violation
scores `0`.

## Objective (minimise)

Let `MSE` be the mean squared error of your expression, evaluated pointwise
on the held-out `rho` set, against the true held-out `mean_hold(rho)`. Let
`nodes` be the number of AST nodes in your expression (a light parsimony
cost). The checker forms

```
Fscore = MSE * (1 + LAMBDA * nodes)
B      = MSE_of(train_mean_hold) * (1 + LAMBDA * 1)     # internal baseline
Ratio  = min(920, 100 * B / Fscore) / 1000
```

with a small fixed `LAMBDA`. Predicting the constant mean of your censored
training hold times exactly reproduces `B` (`Ratio ~= 0.1`). Recovering the
true divergence shape raises the score, but finite-sample noise keeps even a
strong model off a perfect ceiling.

## Why fitting the raw numbers is a trap

Training rows near the top of the training load range are often not really
"hold time = T" — they are "hold time >= T, caller hung up". A regression
fit directly to the logged values (treating every `hold_i` at face value,
capped ones included) is dragged toward `T` exactly where the divergence
signal is strongest, and a smooth low-order curve cannot reproduce a pole
anyway. Extrapolated to near-critical held-out loads, it systematically
**underestimates** the blow-up and predicts a bounded hold time past the
true capacity. The **fraction of rows that got capped**, as a function of
load, is itself informative: it grows smoothly and predictably as `rho`
approaches `rho_max`, and inverting that relationship — instead of
discarding or literally trusting the capped rows — is what actually
localizes the singularity.

## Constraints

`n_train` is 365–500 rows. Time limit 5 s, memory 512 MB. All ten cases use
the same underlying mechanism; only the sampled data, sample size, and how
far the held-out set extrapolates vary with the test id. Scoring is fully
deterministic.
