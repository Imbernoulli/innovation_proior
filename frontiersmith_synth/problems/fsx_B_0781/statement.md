# The Worn Clocktower Gear Train

You are restoring a clocktower's worn gear train. A winding shaft (the
**input**) drives the escapement arbor (the **output**) through a hidden
mesh with two flaws you must separate:

1. **A nominal ratio realised by integer tooth counts.** The output angle is
   `r` times an internal contact angle, `r = p/q` for some **small** positive
   integers `p, q` (real tooth counts) — not an arbitrary real gain.
2. **Backlash.** A hidden dead-band of half-width `D` surrounds the contact
   point: it is dragged along only once the input leaves the current
   `[contact - D, contact + D]` band; inside the band the output is *frozen*
   (unmeshed play). After every direction reversal, slack must be taken up
   again from the opposite edge — the classic hysteretic backlash operator.

You have a logbook from the smith's calibration session: a **slow, mostly
one-directional** turn of the winding key (at most one or two brief
reversals, the smith checking his work). In that quasi-static regime the
backlash lag is taken up almost immediately and stays essentially constant,
so the logged trace looks close to a straight line. You are graded on the
escapement's real behaviour: a **fast** drive reversing direction many
times, fully exposing the hidden state — a memoryless fit to the logbook
extrapolates badly there (see below).

## Input (stdin)

```
n t
x[0]  y[0]
x[1]  y[1]
...
x[n-1] y[n-1]
```

`t` is the test id; `n` training rows follow, each an input shaft angle and
observed output angle (floats, sensor noise included). The held-out grading
drive — a faster, many-reversal trace of the SAME mechanism — is NOT given
to you.

## Output (stdout): a stateful predictor in a tiny DSL

Emit at most two statements:

```
STATE <expr>      (optional; defines the contact register S each tick)
OUT   <expr>       (required; the emitted output angle)
```

The grader **rolls your program forward** over the held-out drive, carrying
one state register `S` across ticks: each tick it evaluates `STATE` (if
present) to get the new `S` (missing history at t=0 defaults to `0.0`; no
`STATE` line ⇒ `S` stays `0` forever), then evaluates `OUT` to produce
`y_hat[t]`.

Expressions are arithmetic over `+ - * /`, parentheses, numeric constants,
unary functions `sig` (logistic), `step` (1 if arg>0 else 0), `relu`, `tanh`,
`absv`, binary functions `min2(a,b)`, `max2(a,b)`, and variables `x` (current
input), `xkJ` (input `J` ticks ago), `S`/`S0` (current state), `SkJ` (state
`J` ticks ago). `STATE` may reference `x`, `xkJ`, `SkJ` (`J>=1`, past state
only — no same-tick self-reference). Delays `J` are `1..24`; the whole
program is `≤ 80` nodes.

**Illustrative FORM only — NOT the hidden mechanism** (different, unrelated
ratio and width; discover the real `p, q, D` from the data):

```
STATE max2( min2( Sk1, x + 0.30 ), x - 0.30 )
OUT   1.5 * S
```

## Feasibility

The program must parse under the grammar above. Any violation, or any
non-finite value produced during rollout, scores `0`.

## Objective (minimise)

Let `MSE` be the mean squared error of your rolled-out `y_hat` against the
true held-out output trace, and `nodes` the number of expression nodes used.
The grader forms

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_constant_0 * (1 + LAMBDA * 1)      # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. Constant-zero reproduces `B` (Ratio ≈ 0.1);
lowering held-out error raises the score, with a light tax on bloat.

## Why the logbook is a trap

On the slow sweep, once the input commits to a direction the contact point
locks onto `x - D` (rising) or `x + D` (falling) and stays there, so
`y ≈ r*x ∓ r*D` is nearly a straight line — a memoryless affine fit
`y = a*x + b` looks excellent (the lag becomes the intercept `b`). On the
fast held-out drive the contact re-crosses every band from BOTH directions
many times a second; a fixed line cannot know which edge is engaged, so
roughly half the ticks sit on the wrong branch of the loop and the error
compounds with every reversal. Recovering the true `r` first — pinning it to
a small rational instead of trusting a confounded real-valued slope — is
what lets you separate the branches and read off `D` cleanly.

## Constraints

Time limit 5 s, memory 512 MB, `n` a few hundred rows. Scoring is fully
deterministic.
