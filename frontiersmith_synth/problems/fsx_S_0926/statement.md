# Reverse-Engineering a Rival Factory's Tool-Wear Recursion

## Problem
A rival machine shop's tool-wear log leaked to you: for a sequence of jobs run
back-to-back on one cutting tool, the log records the idle gap before each
job started, the job's load and material class, and the observed processing
time. Underneath, every job's processing time is driven by a hidden internal
WEAR LEVEL of the tool. The wear level updates **recursively**, once per job,
from its value after the *previous* job:

```
W_i = f(W_{i-1}, gap_i, load_i, material_i)
```

`f` is an unknown recursive law: the tool may partially recover while it sits
idle for `gap_i` time units, and it accumulates wear depending on the job's
`load_i` and `material_i` in some functional form that is **not specified
here** (linear? saturating? some other shape? does recovery even depend on
the gap the way you'd guess?) -- you must infer both the recovery behaviour
and the accumulation behaviour from how consecutive rows of the log relate to
each other. Every job sequence (training log and grading logs alike) starts
a fresh tool at `W_0 = 0` -- they are independent tool histories, not one
continuous timeline.

Once `W_i` is known, the *observed* processing time follows a **given**,
fixed formula:
```
T_i = BASE[material_i] * (1 + ALPHA * W_i) + noise
```
`BASE[0..2]` and `ALPHA` are printed in the input -- you do not need to
discover this half. Your job is to recover the hidden recursion `f`.

**Illustrative FORM only -- not the real law:** the expression
`Wprev * 0.5 + load * 0.01` is a syntactically valid submission (it updates
wear from the previous wear and the load only). It is nowhere near the real
rival factory's law -- real instances have a genuine idle-recovery term and a
load/material-driven accumulation term that only reveal themselves once you
look at how wear estimated at one job relates to wear estimated at the next.

## Input (stdin)
```
t n alpha base0 base1 base2
gap_1 load_1 mat_1 T_1
...
gap_n load_n mat_n T_n
```
`t` is the test id, `n` the number of TRAINING jobs (chronological order),
`alpha`/`base0..2` are the given observation-formula constants above. Row `i`
gives the idle gap before job `i` (`gap_1 = 0`), the job's load (a positive
real "load units" figure), its material class (`0`, `1`, or `2`), and the
noisy observed processing time `T_i`.

## Output (stdout)
Exactly **one line**: a single arithmetic expression string using only the
variables `Wprev`, `gap`, `load`, `m0`, `m1`, `m2` (one-hot material flags),
`idx` (the job's 1-based position within its own sequence), the operators
`+ - * / **` and parentheses, numeric literals, and the functions `exp log
sqrt abs min max tanh`. No other names, statements, or lines are allowed.

## Feasibility
The output must be exactly one line; it must parse as a single arithmetic
expression using only the allowed names/operators/functions (else
`Ratio: 0.0`); and, when rolled forward on the grading sequences below, it
must never produce a non-finite value at any step (else `Ratio: 0.0`).

## Objective (minimize)
Your expression is treated as the recursive law. The grader regenerates
several HELD-OUT job sequences -- five times longer than training, with a
different load / material / idle-gap mix than training -- under the TRUE
hidden recursion. It rolls your expression forward on the same sequences
(fresh `Wprev = 0`, `idx = 1` at the start of each), turning your predicted
`W_i` into a predicted `T_i` via the given observation formula, and computes
the mean squared error `F` between your predicted `T` and the true (noisy)
held-out `T`, times a small surcharge for the number of operators/function
calls `ops` in your expression.

## Scoring
```
F = heldout_MSE(your expression) * (1 + 0.004 * ops)
B = heldout_MSE(no-wear baseline, i.e. always predict W = 0)
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Submitting `0` (wear stays at exactly zero forever) reproduces `B` exactly,
scoring `Ratio = 0.1`. Recovering the true recursion drives `F` well below
`B`; sensor noise and the surcharge keep `Ratio = 1.0` out of reach.

## Constraints
`1 <= t <= 10`; `20 <= n <= 74` training jobs; time limit 5s, memory 512m;
each `.in` file is well under 5 MB.

## Example (worked score)
Suppose the checker's no-wear baseline gives `heldout_MSE = 0.90` (`B =
0.90`). A submission with 12 operators achieves `heldout_MSE = 0.15`, so
`F = 0.15 * (1 + 0.004*12) = 0.1572`. Then `sc = 100 * 0.90 / 0.1572 =
572.6`, giving `Ratio = 0.5726`.
