# Delivery Delays That Feed On Themselves

A regional distributor logs, every period, the **orders** `O` it places with its
supplier and the **lead time** `L` it observes before delivery. Two facts about
this supplier are on file: its declared **capacity** `Cap` (max orders/period it
can process) and its declared **free-flow lead time** `L0` (the delay with an
empty queue). Your job: give a formula that predicts `L` from what the
distributor can measure.

There is a feedback loop baked into the data. When last period's lead time ran
above `L0`, buyers get nervous and inflate this period's order (they order
earlier and larger to protect against the delay) — which pushes the supplier's
queue up, which raises the *next* lead time, which inflates orders further.
During the logged window demand stayed comfortably below capacity, so this loop
only ever produced small ripples: `O` wobbles a little, `L` wobbles a little, and
the relationship between them looks nearly flat. You will be graded on a
**held-out demand-shock window** — a period of sustained higher demand — where
the same loop settles at a **materially higher, still-orderly** utilization
level that the logged window never approached.

**Illustrative FORM only — NOT the hidden law** (just the allowed syntax):
`0.4 + 0.02*O - 0.01*D`. The real relationship has a different shape and you
must discover it from the data.

## Input (stdin)
```
n t Cap L0
D[0] O[0] L[0]
D[1] O[1] L[1]
...
```
`n` training rows follow; `t` is the test id. Each row: that period's raw demand
estimate `D`, the order actually placed `O`, and the observed lead time `L`.

## Output (stdout)
One line: a closed-form Python expression for `L` in the variables `O`, `D`,
`Cap`, `L0`. Allowed: `+ - * / **`, unary `-`, numeric constants, and the
functions `sqrt log exp sig tanh absv`. No other names.

## Scoring (deterministic, maximisation)
Your expression is evaluated at each row's `O` (with `D`, `Cap`, `L0` bound too)
on a **held-out shock window**, regenerated inside the grader from the test id —
never shown to you. Let `p_i` be your prediction and `t_i` the true held-out lead
time:

```
err_i    = |ln(p_i / t_i)|                        (p_i<=0 counts as a huge error)
quality  = mean_i  1 / (1 + 6 * err_i)
F        = quality * (1 / (1 + LAMBDA * nodes))    # nodes = expression size
baseline = same quality formula for the constant predictor mean(L_train)
Ratio    = min(1000, 100 * F / baseline) / 1000
```

Higher `Ratio` is better (capped at `1.0`). A constant predictor scores about
`0.1`. `LAMBDA` is a small fixed parsimony weight. Non-finite predictions, or a
non-finite/disallowed expression, score `0`.

## Why the calm-window curve is a trap
Over the logged window `O` only ranges across a narrow band well below `Cap`, so
a plain best-fit LINE through the `(O, L)` cloud looks excellent — R² near 1.
But the true relationship has a **pole at `O = Cap`**: as orders approach
capacity the queue — and hence the lead time — blows up faster than any line
can track. A line calibrated on the calm band systematically **underpredicts**
once the shock window pushes `O` well past where you trained, exactly where it
is graded.

## Constraints
- Time limit 5 s, memory 512 MB. `n` is at most a few hundred rows.
- `Cap`, `L0` are always positive; `L` values in training are always `> L0`.
- Scoring is fully deterministic; irreducible noise on the shock window means
  even a correctly-shaped formula does not reach `Ratio = 1.0`.
