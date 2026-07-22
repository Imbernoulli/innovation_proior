# Proofing Box Crossover -- recovering a hidden second kinetic channel from Arrhenius-plot curvature

A bakery logs the **dough rise rate** `r` (arbitrary rate units) of one
dough/yeast-strain batch against the **proofing-box temperature** `T`
(Kelvin), sweeping only the **warm, mid-range window** a proofer can hold
steady (roughly 25-42 C). One batch obeys a fixed but undocumented law:
`r(T)` combines **two competing exponential channels** -- a fermentation
channel whose rate *rises* with temperature, and a stability channel whose
rate *falls* as temperature rises -- the way two resistors combine **in
series**: whichever channel is momentarily slower bottlenecks the observed
rate. Reverse-engineer a closed-form law `r(T)` that stays correct
**outside** the proofing window, including where the rise **stalls** as the
oven overshoots into heat.

The catch: inside the window the fermentation channel is faster everywhere,
so a single best-fit exponential (an Arrhenius-plot straight line) tracks
the measurements to the noise floor. You are graded on a **held-out grid**
reaching well below the window (a cold fridge-retard regime) and well above
it (an oven-overshoot regime), where the stability channel has become the
bottleneck and the true curve turns over and stalls, far from any single
exponential's monotone trend.

## Input (stdin)

```
t  N
T_0  r_0
T_1  r_1
...
T_(N-1)  r_(N-1)
```

`t` is the test id. Then `N` measurement rows follow (already sorted by
`T`), each a temperature (Kelvin) and its measured rise rate. The held-out
grading grid (both the cold and the hot regimes) is **not** given to you.

## Output (stdout): a closed-form law

Emit a single expression for `r` as a function of `T`. Allowed: numeric
constants, the operators `+ - * /`, unary `+/-`, parentheses, the single
variable `T`, and the functions `absv(a)`, `minv(a,b)`, `maxv(a,b)`,
`powv(a,b)` (computes `a` to the power `b`; `a` must evaluate positive),
`expv(a)` (computes `e^a` for any finite `a`).

**Illustrative FORM only -- NOT the hidden law:**

```
2.5 + 0.3*absv(T - 306.0) / (1.0 + 0.01*T)
```

This only shows the syntax; the real law's shape, channels and coefficients
are different and must be discovered from the data.

## Feasibility

The expression must parse under the grammar above (only known names/
functions, correct arities, finite constants, at most 200 expression nodes).
Any parse violation, or any non-finite or non-positive value produced while
evaluating the law on the grading grid, scores `0` for that test.

## Objective (minimise)

Let `pred_k` be your law evaluated at held-out temperature `T_k`, and
`true_k` the (noisy) true rise rate there. The grader forms the mean
**squared LOG error** (it rewards matching the turnover, not just the scale)
plus a small parsimony tax on expression size `nodes`:

```
F = mean_k (log(pred_k) - log(true_k))^2 * (1 + LAMBDA * nodes)
B = mean_k (log(rbar)   - log(true_k))^2 * (1 + LAMBDA * 1)   # rbar = flat
                                            # geometric mean of your OWN
                                            # training r values
Ratio = min(0.90, 0.1 * (B / F) ** GAMMA)
```

with small fixed constants `LAMBDA, GAMMA` (`0 < GAMMA < 1`), capped below 1
so the score never saturates. Predicting the flat training average
reproduces `B/F = 1` (Ratio = 0.1); a law with the right turnover drives `F`
down and pushes `B/F` above 1, raising the Ratio. The sub-linear exponent
`GAMMA` keeps a merely-correct-shaped law from saturating even though `B/F`
can span a wide range once the turnover is close. Measurement noise and the
finite training sample keep even a strong law below the ceiling -- report
the highest Ratio you can.

## Why the proofing window is a trap

Inside the window the fermentation channel is so much faster that a single
exponential fits the measurements beautifully -- nothing in-sample screams
that a second channel exists. But the stability channel is falling toward it
across that same window, leaving a small, systematically **curved** residual
on the Arrhenius plot (log r vs 1/T): concave, bending away from the
best-fit line at *both* ends. That curvature -- easy to write off as noise
-- is the only in-window evidence that a second channel exists at all. Take
it seriously: test whether it is real curvature or noise, and if real,
exploit the two channels' **series (harmonic-mean) structure** rather than
fitting one blended exponential.

## Constraints

Time limit 5 s, memory 512 MB. `N = 120` rows; scoring is fully
deterministic.
