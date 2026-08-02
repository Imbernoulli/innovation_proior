# Queue Saturation Forecast

A single-server queueing system has a hidden **capacity** `C`: as the offered
load `L` climbs toward `C`, the mean wait time grows without bound. Below
capacity, wait time also scales with the **burstiness** `B` of arrivals — a
squared-coefficient-of-variation-style covariate (`B=1` roughly Poisson,
`B>1` bursty/clumped, `B<1` regular/smooth).

You are handed a log recorded while the system ran under **low-to-moderate**
offered load — comfortably below capacity. In that sub-saturation range, wait
time looks almost linear in `L`; the eventual blow-up shows up only as a
**mild curvature**. You will be graded on **higher offered loads** you never
observed, where that curvature — invisible if you don't look for it —
dominates.

## Input (stdin)

```
n t
L[0]  B[0]  W[0]
L[1]  B[1]  W[1]
...
L[n-1] B[n-1] W[n-1]
```

`t` is the test id. `n` training rows follow: offered load `L` (positive
float), burstiness `B` (positive float, roughly in `[0.2, 2.2]`), and the
measured mean wait `W` (positive float, with measurement noise). The
held-out grading loads are a **higher**, non-overlapping load range for the
*same* hidden system; they are NOT given to you.

## Output (stdout): one closed-form expression

Print **one line**: an arithmetic expression over `+ - * / **`, parentheses,
numeric constants, and the variables `L` and `B` only (no function calls, no
other names). It is evaluated directly as your predicted wait time at each
held-out `(L, B)`.

**Illustrative FORM only — NOT the hidden law:**
```
0.4 * L + 0.1 * B - 0.02 * L * B
```
This just shows the syntax; the real relationship has a different shape you
must discover from the data.

## Feasibility

The expression must parse under the grammar above (only names `L`, `B`;
finite numeric constants; at most 60 expression nodes). Evaluating it at
every held-out point must produce a finite, non-negative result — wait times
cannot be negative. Any violation scores `0`.

## Objective (maximise)

Let `MSE` be the mean squared error of your expression's predictions against
the held-out wait times, and `nodes` the number of expression nodes. The
grader forms

```
F = MSE * (1 + 0.01*nodes)
B = MSE_of_k*L * (1 + 0.01*3)      # internal baseline: best proportional fit on your training rows, ignoring B
r = B / F
Ratio = 0.88 * r / (r + 7.8)
```

A submission that reproduces the baseline scores `Ratio ~= 0.10`. Lowering
held-out error raises the score, but the mapping is a bounded curve: no
matter how good `F` gets, `Ratio` cannot reach `0.88` — held-out measurement
noise, plus the fact that the true burstiness exponent is only *approximately*
2, keep a perfect fit out of reach. A needlessly large expression is taxed
via `nodes`.

## Why the low-load data is a trap

Fit any linear-in-`L` model — even one that gets the burstiness weighting
exactly right — to the sub-saturation rows, and it will look excellent
there: the curvature from the approaching capacity limit is small at low
utilization. But a purely linear model has no pole; extrapolated to higher
load it silently *underestimates*, and the gap widens the closer the
held-out load gets to true capacity. The training curvature — however
faint — is the only clue to where that capacity actually is.

## Constraints

Time limit 5 s, memory 512 MB. `n` is 50–70 rows. Scoring is fully
deterministic.
