# Merge-Junction Delay: Learn the Jam Law from Light Traffic Only

## Problem

A monitored link (link `0`) sits just downstream of a junction where two
feeder links (`1` and `2`) partially merge into it — some of their traffic
spills over and adds to link `0`'s own queue. A sensor logs the flow on all
three links plus the resulting travel delay on link `0`.

Delay grows the way road congestion always does: **superlinearly** in an
*effective combined load* — link `0`'s own flow plus a fraction of each
feeder's flow that actually merges through. The exponent governing that
superlinear growth is **shared** across the whole junction (it is a
property of the road surface / driver behaviour, not of any one link), but
the two feeders' spillover fractions differ from each other and are not
given to you.

You are handed sensor logs recorded **only during a light-traffic window**
(all three flows modest, well below capacity). Your job is to output a
closed-form expression for the delay that **extrapolates** correctly to a
**heavy-traffic** regime you never observed — including combinations where
one link is light while its neighbours are heavy, and vice versa.

## Input (stdin)

```
N
x0 x1 x2 y      (row 1)
...             (N rows total)
```
`x0, x1, x2` are the three link flows (positive floats, roughly in `[0, 3]`
in this light-traffic log); `y` is the measured delay (positive, noisy —
loop-detector telemetry is not clean). `N` grows with the difficulty ladder
(60 to 114 rows).

## Output (stdout)

One line: a closed-form Python expression in `x0, x1, x2`. Allowed tokens
only:
- arithmetic: `+ - * / ** %`, parentheses, numeric literals;
- unary functions (exactly one argument each): `exp, log, sqrt, sin, cos, tanh, abs`;
- constants `pi`, `e`.

Any other name, attribute access, indexing, or multi-line output is
rejected. At most 400 AST nodes, at most 5000 characters, single line.

## Feasibility

The expression must parse under the whitelist above and evaluate to a
**finite** real value of moderate magnitude (`|value| <= 1e6`) at every
held-out point. Anything else (empty, unparseable, unknown symbol,
`nan`/`inf`, absurd magnitude) scores **0**.

## Objective (minimize)

The grader regenerates a deterministic **held-out HEAVY-traffic split** —
flows several times larger than anything you trained on, including
corner combinations that isolate the own-link effect from the feeder
(coupling) effect — evaluates your expression there, and forms the loss

```
F = RMSE_heldout + ALPHA * complexity        (ALPHA = 0.004, complexity = # AST nodes)
```

so raw accuracy trades against expression simplicity: there is no free
lunch for a huge formula that merely memorizes the light-traffic rows.

## Scoring

Let `B` be the held-out RMSE of the internal constant predictor
(`y = mean(train y)`). Then

```
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```

Reproducing the constant baseline gives `Ratio ≈ 0.1`. Recovering the true
shared exponent and the two feeder couplings drives the held-out RMSE down
and pushes the ratio well above that — but heavy measurement noise on the
light-traffic log and the complexity penalty keep a perfect score out of
reach.

## Constraints

- `60 <= N <= 114` training rows.
- Training flows lie roughly in `[0, 3]`; held-out (heavy-traffic) flows lie
  roughly in `[4, 7]`, a genuinely disjoint, larger region, with several
  points specifically chosen so one link is light while its neighbours are
  heavy (and vice versa).
- Deterministic scoring only; the held-out region and ground-truth law are
  fixed per instance and regenerated inside the grader — no wall-time, no
  randomness in the score.

## Example (illustrative FORM only — NOT the hidden law)

Suppose (hypothetically) the three links did not interact at all and delay
were just an independent sum, e.g. `y = 4.0 + 0.3*x0 + 0.1*x1*x2`. A
submission in that shape would be written as:

```
4.0 + 0.3*x0 + 0.1*x1*x2
```

This shows only the required output format and allowed tokens — the actual
delay law involves a shared power-law exponent over a combined load and
must be discovered from the data, not pattern-matched from this example.
