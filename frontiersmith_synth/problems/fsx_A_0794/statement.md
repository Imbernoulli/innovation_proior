# Assembly Stack-Up Variability — the batch you shipped it in matters

A precision-assembly logbook records the total end-to-end positional
deviation `D` of stacked mechanical assemblies made of `n` components. Each
component `i` contributes an independent manufacturing tolerance `sigma_i`
(a standard deviation). `S2 = sum_i sigma_i^2` is the familiar
**root-sum-square (RSS)** variance every tolerance-stacking textbook uses,
assuming every part's error is independent of every other part's.

But components are cut and shipped in **batches** (production lots): a run
of `n` parts is split into `m` batches of sizes `b_1..b_m` (`sum b_j = n`).
Same-batch parts share a small common-mode bias — same tool wear, same
raw-material lot, same shift — which correlates their deviations. A batch of
size `b` contributes on the order of `b^2` worth of correlated variance
(all `b` parts drift together), versus only `b` if independent. This gives a
second structural summary, `B2 = sum_j b_j^2`, capturing how much variance
comes from same-batch correlation — depending on batch *layout*, not just
`n` or `S2`.

The hidden law has the shape `D = sqrt(S2 + TAU2 * B2)` for some fixed hidden
coefficient `TAU2 > 0` (units of `sigma^2`), corrupted by multiplicative
measurement noise. Recover a closed-form expression for `D`.

## Input (stdin)
- Line 1: two integers — the row count and the test id.
- Next rows: `n m S2 B2 D`, one recorded assembly per line (`n`, `m`
  integers; `S2`, `B2`, `D` floats; `D > 0`).

The recorded logbook is from a **short-run pilot line**: assemblies are
small (`n` from 3 to 12) and shipped in **small batches** — every batch has
at most 3 parts — so `B2` stays roughly proportional to `n`, and the
correlated term `TAU2*B2` is a small fraction of `S2`, comparable to (or
below) the measurement noise.

## Output (stdout)
One line: a closed-form Python expression for `D` in the variables `n`, `m`,
`S2`, `B2`. Allowed: `+ - * / **`, unary `-`, numeric constants, and the
functions `sqrt log exp sig tanh absv`. Example (illustrative **form only —
NOT the hidden law**): `0.5*n + log(S2 + 1.0) - absv(m - B2)`. No other names
are accepted.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out extrapolation log**, regenerated
inside the grader: much LONGER assemblies shipped in a HANDFUL of LARGE
batches (batch count stays small as `n` grows, so `B2` scales up roughly
like `n^2` there instead of `n`). Let `p_i` be your prediction and `t_i` the
true (noisy) deviation at held-out row `i`:

```
metric   = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor geomean(train D)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
scores about `0.1`. `LAMBDA` is a small parsimony weight, so an overgrown
expression is penalized. Non-finite predictions score `0`.

## Why the obvious fit is a trap
The textbook approach to tolerance stacking is RSS: assume all parts are
independent and predict `D = sqrt(S2)` (perhaps with a fitted scale
correction). This fits the pilot-line logbook nicely — batches are tiny, so
the correlated term is a small fraction of `S2`, buried in sensor noise — and
nothing in a naive curve fit forces you to notice `B2`, let alone give it its
own additive term. On the held-out log, where production has moved to a
handful of large batches, `B2` overtakes `S2` and the RSS prediction falls
far short of the truth.

Recovering the correlated term means hypothesizing the *composition*: total
variance is independent-part variance PLUS a same-batch correlation term,
`D^2 = S2 + TAU2*B2`, not a single power law in `n` or `S2`. Fit `TAU2` from
the residual `D^2 - S2` regressed against `B2` across the pilot-line rows —
the coefficient is small and easy to miss row-by-row, but averaging the
regression over many rows recovers it, and it is exactly what extrapolates to
the large-batch regime.

## Constraints
- Time limit 5 s, memory 512 MB; row count up to a few hundred.
- Held-out noise leaves irreducible error, so even a correct law does not
  reach `Ratio = 1.0` — there is room above the reference solutions.
