# River Flow After the Snow Is Gone

You are modelling the outflow of a small mountain watershed. Each tick you get
the day's **precipitation** `p` (non-negative, normalised) and **temperature**
`tm` (normalised; zero is roughly freezing, negative cold, positive warm). The
watershed responds with a **flow** `y`.

Physically: when cold, precipitation falls as snow and piles up in a
snowpack. When warm, that snowpack melts at a rate set by how far above
freezing it is (capped by however much snow is left); both the melt and any
rain falling while warm add to the flow. Recent rain and meltwater also leave
the ground temporarily wetter, and wetter ground sheds new rain into flow more
efficiently — an antecedent-moisture effect fading over a few ticks.

The training log was recorded during a **mild, rain-dominated** stretch:
mostly above freezing, with only a few brief cold snaps. There the snowpack
barely accumulates, so flow looks almost like a direct, memoryless reaction to
recent rain. You will be graded on a **held-out winter-into-spring** stretch —
many quiet cold weeks that build a real snowpack, then a long warm stretch
where that stored snow (not today's rain) drives the flow, including
stretches with zero rain.

## Input (stdin)

```
n t
p[0]  tm[0]  y[0]  proxy[0]
p[1]  tm[1]  y[1]  proxy[1]
...
```

`t` is the test id; `n` training rows follow: precipitation, temperature,
flow, and a **noisy snow-depth proxy sensor** reading (floats). `proxy` hints
that a storage mechanism exists — it is never given on the held-out stretch.

## Output (stdout): a stateful predictor in a tiny DSL

Emit at most two statements:

```
STORE <accum_expr>     (optional; at most one -- updates ONE storage register)
OUT   <out_expr>        (required; the emitted flow value)
```

The grader **rolls your program forward** over the held-out stretch, carrying
one register `SW`: each tick it evaluates `accum_expr` (if `STORE` is
present) and sets `SW = clip(SW_prev + accum_expr, 0, 8.0)` (no `STORE` ⇒ `SW`
stays `0`); then `OUT <out_expr>` produces `y_hat`.

Expressions are arithmetic over `+ - * /`, parentheses, numeric constants, the
unary functions `sig` (logistic), `step` (1 if arg>0 else 0), `relu`, `tanh`,
`absv`, and these variables:

- `p` — today's precipitation; `pkJ` — precipitation `J` ticks ago.
- `tm` — today's temperature; `tmkJ` — temperature `J` ticks ago.
- `SW` (=`SW0`) — the register **after** this tick's update (`OUT` only);
  `SWkJ` — the register `J` ticks ago.

`STORE`'s `accum_expr` may use `p`, `tm`, their delayed taps, and `SWkJ`
(`J>=1`) — **never** `SW`/`SW0` itself (that would be self-reference). Delays
`J` are `1..24`; the whole program is `≤ 140` nodes.

**Illustrative FORM only — NOT the hidden law:**

```
STORE p - 0.4 * relu ( tm )
OUT   0.05 + 0.3 * relu ( p - pk2 ) + 0.2 * SWk3
```

This shows the syntax only; the real mechanism differs and must be discovered
from the data.

## Feasibility

The program must parse under the grammar above (known names/functions only,
finite constants, delays/size within bounds). Any violation, or any
non-finite value during rollout, scores `0`.

## Objective (maximise predictive skill)

Let `MSE` be the mean squared error of `y_hat` against the true held-out flow,
and `nodes` the total expression-node count. The grader forms

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_constant_mean(train_flow) * (1 + LAMBDA * 1)   # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. Predicting the training mean reproduces `B`
(Ratio ≈ 0.1); lower held-out error raises the score, with a mild parsimony
tax on needlessly large programs. Report the highest Ratio you can.

## Why the training season is a trap

On training data, snow rarely accumulates, so a curve fit purely to `p`/`tm`
(plus a short lag window standing in for antecedent moisture) already tracks
flow well — the snowpack term is nearly always zero there. The held-out
accumulation phase runs far longer than any allowed lag (`J ≤ 24`), so no
window of recent `p`/`tm` can substitute for a running total — only a
register persisting across the whole cold season can carry the stored water
forward to where it drives the later melt-dominated flow, including after the
snow finally runs out.

## Constraints

Time limit 5 s, memory 512 MB. `n` is a few hundred rows. Scoring is fully
deterministic.
