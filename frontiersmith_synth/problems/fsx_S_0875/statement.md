# Drone Fleet Drag

Your delivery drones fly slower than commanded whenever other drones crowd
their sensing radius. Nobody wrote down the mechanism. You have logs from
flight tests: for each drone at an instant, its **commanded speed**, its
**realized speed**, and the **distances** to every other drone currently
inside its sensing radius. Your job: reverse-engineer a predictor of realized
speed from commanded speed and neighbor geometry.

The logs come from three kinds of flights, distinguishable only by their
neighbor count and spacing pattern (no label is given): **isolated-pair**
passes (exactly one neighbor, swept across a range of separations),
**controlled cluster holds** (a small ring of drones held at a fixed,
near-uniform spacing while the ring size is varied), and ordinary **organic**
mixed-fleet flights. Every logged flight ever involved only 3-8 drones.

You will be graded on **held-out swarms of 20-40 drones flown roughly three
times denser** than anything in the logs. Predictors that merely curve-fit
the logged regime tend to extrapolate badly there.

## Input (stdin)

```
n_rows t
n1 v1 y1 d1_1 d1_2 ... d1_n1
n2 v2 y2 d2_1 ...
...
```

`t` is the test id. Each of the `n_rows` lines is one drone's reading: `n`
(neighbor count), `v` (commanded speed), `y` (realized speed), then `n`
neighbor distances (all positive floats).

## Output (stdout): a two-statement program

```
KERNEL <expr over "dist">
OUT    <expr over "S" and "v">
```

The grader evaluates `KERNEL` on every neighbor distance of a graded row,
sums the results into `S`, then evaluates `OUT` with that `S` and the row's
commanded speed `v` to get the predicted realized speed. Expressions use only
`+ - * /`, parentheses, unary minus, numeric constants, and the one named
variable legal in that statement (`dist` for `KERNEL`; `S` and `v` for
`OUT`) — no function calls, no other names.

**Illustrative FORM only — NOT the hidden law:**

```
KERNEL 0.4 * dist
OUT    v - 0.1 * S * S
```

This just shows the syntax (and is a bad model of anything); the real
mechanism has a different shape you must discover from the data.

## Feasibility

The program must be exactly one `KERNEL` and one `OUT` statement, parsing
under the grammar above (known names only, finite constants, ≤ 50 expression
nodes total). Any violation, or any non-finite value produced while scoring,
scores `0`.

## Objective (maximize)

Let `MSE` be the mean squared error of your predicted realized speeds against
the true held-out values. The grader forms a bounded goodness score
`F = 1 / (1 + MSE)` and compares it to `B`, the same goodness measure for the
baseline "ignore interference, predict realized = commanded":

```
Ratio = min(1000, 100 * F / B) / 1000
```

Reproducing the baseline scores `Ratio ~= 0.1`. Cutting held-out error
sharply raises the score; measurement noise and estimation error keep even a
strong recovery below the ceiling.

## Why the logged regime is a trap

Within the logged 3-8-drone flights, neighbor *count* and true interference
happen to correlate well enough that "more neighbors, more slowdown" looks
like the whole story — a saturating curve fit against count alone tracks the
training rows convincingly. But the held-out swarms are not just bigger, they
are flown roughly three times **denser**: the same neighbor count now sits
much closer together, so each neighbor's true pairwise contribution is far
larger than anything the count-only recipe was calibrated on. A model that
never separated "how many" from "how close" has no way to see that coming and
systematically under-predicts the slowdown. Recovering the actual
distance-dependent kernel — and the bounded shape it feeds into — requires
treating the isolated-pair and controlled-cluster rows as the designed
sub-experiments they are, not just more rows to pool into one count-based fit.

## Constraints

Time limit 5 s, memory 512 MB. `n_rows` is at most a few hundred; each row has
at most 7 neighbors. Scoring is fully deterministic.
