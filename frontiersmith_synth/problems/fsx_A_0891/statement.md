# The Retired Engineer's Intersection Controller

A retired traffic engineer built deterministic intersection controllers for a
small town. Each controller watches a junction's approaches and, once per
cycle, picks exactly ONE approach to receive the green light. Nobody kept the
source code, only the data logger. You have logs from several junctions.

Every logged cycle records, for each approach `i` of that junction:

- `q` — vehicles currently queued on that approach.
- `w` — the approach's static weight (lane count / arterial importance;
  fixed for the whole junction).
- `a` — cycles elapsed since that approach last got the green (its "age").
- `cw` — how many steps clockwise the approach sits from whichever approach
  got the PREVIOUS green (`0` = the very next approach clockwise; if there
  was no previous cycle, clockwise is measured from a fixed reference).

...and which approach actually got the green that cycle. The controller
combines `q`, `w`, `a` and clockwise order into its choice — including,
possibly, some special-case rule to stop an approach waiting forever — but
the exact combination and any such rule are unknown. You must induce them
from the traces.

**Crucial catch — topology split.** Every logged junction here has exactly
**3 or 4** approaches. You will be graded on junctions with **5 or 6**
approaches — degrees that never appear in your log. A rule that only fits
patterns specific to 3-way/4-way junctions (a lookup table, a per-position
special case) will not transfer; a rule that captures the controller's
actual abstract logic will.

## Input (stdin)

```
TESTID <t>
NSTATES <m>
STATE N=<n> LASTGREEN=<lg> WINNER=<w_idx>
Q <q_0> <q_1> ... <q_{n-1}>
W <w_0> <w_1> ... <w_{n-1}>
A <a_0> <a_1> ... <a_{n-1}>
CW <cw_0> <cw_1> ... <cw_{n-1}>
```

`m` such `STATE` blocks follow, each an independent junction snapshot: `n`
approaches, which approach won (`w_idx`), and the four per-approach arrays
above (index `i` matches across `Q`/`W`/`A`/`CW`). `LASTGREEN` is only
context (`-1` if there was no previous cycle at that junction).

## Output (stdout): one priority expression

```
PRIORITY <expr>
```

`expr` is a single arithmetic expression over the per-approach variables
`q, w, a, cw, n` (numeric constants, `+ - * /`, parentheses), plus the unary
functions `sig` (logistic), `step` (`1` if arg`>0` else `0`), `relu`,
`tanh`, `absv`. At most 60 expression nodes.

**Illustrative FORM only — NOT the hidden law:**

```
PRIORITY tanh ( q - 2 * w ) + 0.1 * relu ( a - cw )
```

This only shows the syntax; the real controller's rule has a different
shape and you must discover it from the data.

## Grading

The grader regenerates a HELD-OUT set of 5-way and 6-way junction episodes
from the same family of controllers (never shown to you), and for every
snapshot evaluates your expression once per approach (substituting that
approach's own `q, w, a, cw` and the junction's `n`). The approach with the
**largest** value is your prediction (ties broken by the lower approach
index — if you want the controller's own tie-break convention you must
encode it yourself as a tiny distinguishing term in `expr`). Your score is
prediction **accuracy** on the held-out set, lightly discounted by
expression size, normalized against the checker's own baseline predictor
`PRIORITY 0` (a permanent tie, i.e. "always guess approach 0"):

```
F = accuracy / (1 + LAMBDA * nodes)
B = baseline_accuracy / (1 + LAMBDA * 1)
score = min(10, F / B) / 10        (so B itself scores ~0.1)
```

## Feasibility

Output must parse as exactly one `PRIORITY <expr>` line using only the
named variables/functions above, with finite constants, within the node
budget. Any violation, or any non-finite value produced while evaluating
`expr`, scores `0`.

## Worked example (toy, not the real controller or its constants)

Suppose a held-out set has 4 states and your submitted expression predicts
the winner correctly on 3 of them (`accuracy = 0.75`), your expression has
6 nodes, and the baseline `PRIORITY 0` gets `baseline_accuracy = 0.20` on
the same 4 states (1 node, by definition). With `LAMBDA = 0.006`:
`F = 0.75 / 1.036 ≈ 0.724`, `B = 0.20 / 1.006 ≈ 0.199`, ratio
`≈ min(10, 3.643)/10 = 0.364`.

## Constraints

`3 ≤ n ≤ 6` in any state you may see or be graded on; `0 ≤ q ≤ 40`;
`1 ≤ w ≤ 3`; `0 ≤ a ≤ 40`; `0 ≤ cw < n`; `1 ≤ m ≤ 2000` training states.
Time limit 5s, memory 512MB.
