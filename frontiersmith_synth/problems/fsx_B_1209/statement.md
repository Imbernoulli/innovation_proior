# Permafrost Thaw Forecast

A remote ground station logs, at each tick, a normalised **surface energy
forcing** `f[t]` (positive = net heat flowing into the ground, negative = net
heat flowing out) and a **ground thermal index** `G[t]`. Your job: forecast
`G` forward.

Underneath, the station tracks a hidden **energy accumulator**: a running sum
of `f`, floored at zero (a cold snap can only drain surplus heat back down to
zero — it can't create a "heat debt"). While that accumulator stays below a
hidden **latent-heat capacity**, `G` is pinned at a **frozen baseline**, almost
independent of how `f` itself wobbles — the classic Stefan-problem plateau: melting
ice buffers the temperature until it's gone. Only once the accumulator
crosses the capacity does an **active (thawed) layer** start growing, roughly
as the **square root of the excess energy** beyond the capacity, and an
**insulation-loss feedback** (exposed ground absorbs more of each unit of
forcing than moss/snow-covered ground did) makes further forcing count for
more once thaw is underway. The layer's depth **saturates** at a bounded
maximum (it can't outgrow the ground itself). `G` then equals the frozen
baseline plus a scaled, noisy reading of that depth.

The logs you are given were recorded on a window where the station stayed on
the frozen plateau the whole time — `G` looks like flat noise around a
constant, seemingly insensitive to whatever `f` is doing. You will be graded
on a **longer continuation** of the same forcing process (never shown to
you), which may cross into the thaw regime.

## Input (stdin)

```
n t
f[0]  G[0]
f[1]  G[1]
...
f[n-1] G[n-1]
```

`t` is the test id; `n` training rows follow. The graded trace continues the
same station for longer; it is NOT given to you.

## Output (stdout): a stateful predictor in a tiny DSL

Emit at most two statements:

```
ACC <expr>      (optional; at most one — your own energy-accumulator register)
OUT <expr>       (required — the emitted ground-index prediction)
```

The grader **rolls your program forward** over the graded window, carrying
one accumulator register `A` (initially `0`, no memory of training):

- Each tick it evaluates `ACC`'s expression and stores the result as the new
  `A`. (No `ACC` line ⇒ `A` stays `0` forever.)
- Then it evaluates `OUT` to produce the prediction.

Expressions are arithmetic over `+ - * /`, parentheses, numeric constants,
the unary functions `sig` (logistic), `step` (1 if arg>0 else 0), `relu`,
`tanh`, `absv`, `sqrt` (of a non-negative value only), and these variables:

- `f` — the current forcing; `fkJ` — forcing `J` ticks ago (e.g. `fk3`).
- `A` (=`A0`) — the current accumulator; `AkJ` — accumulator `J` ticks ago.

`ACC`'s own expression may reference `AkJ` for `J>=1` (the accumulator's OWN
past — this is how you build a running integrator) but never `A`/`A0` itself.
Delays `J` must be `1..24`; the whole program must be `≤ 80` nodes.

**Illustrative FORM only — NOT the hidden law:**

```
ACC relu ( Ak1 + fk2 - 0.1 )
OUT 0.3 + tanh ( A - 1.5 )
```

This just shows the syntax; the real law is different and must be discovered
from the data — training alone never shows you a crossing.

## Feasibility

The program must parse under the grammar above (known names/functions,
finite constants, delays and size in bounds). `sqrt` of a negative value, or
any non-finite value produced during rollout, scores `0`.

## Objective (maximise)

Let `MSE` be the mean squared error of your rolled-out prediction against the
true graded `G`, and `nodes` the expression-node count of your program:

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_constant_0.0 * (1 + LAMBDA * 1)      # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. Predicting the constant `0.0` reproduces `B`
(Ratio ≈ 0.1); lowering graded error raises the score, with a small parsimony
tax against bloated programs.

## Why the flat training window is a trap

A regression fit to the visible `G` alone sees noise around a constant and
correctly concludes "nothing is changing" — for the training span. But `f`
itself may carry a slow drift invisible in `G` (the plateau absorbs it). A
predictor that only ever looks at `G`'s own history has no way to notice this
and will confidently extrapolate continued stability into the graded window —
precisely when, on several of the ten stations, the accumulator finally
crosses its capacity and `G` climbs. Tracking the accumulated *energy*, not
the *temperature*, is what sees it coming.

## Constraints

Time limit 5 s, memory 512 MB. `n` is a few hundred rows. Scoring is fully
deterministic.
