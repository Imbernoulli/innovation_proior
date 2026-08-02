# Batch Throughput Past the Knee

A fixed inference kernel is served on one accelerator at growing batch size.
Someone benchmarked it, but only at **small** batches — anything larger risked
an out-of-memory kill on the shared box. You are handed those measurements
plus the kernel's cost accounting, and must forecast throughput at batch
sizes nobody has dared try yet.

You also know two hardware ceilings for this device: its peak compute rate
`C` (FLOPs/s) and its peak memory bandwidth `W` (bytes/s). You know the
kernel's per-sample cost: `F` FLOPs and `D` bytes of traffic per sample.
Throughput (samples/s) at batch size `x` can never exceed **either** ceiling
converted to a per-sample rate: `C/F` samples/s if compute is the limiter, or
`W/D` samples/s if memory bandwidth is the limiter — whichever is tighter for
this kernel. Below that ceiling, throughput climbs with batch size (more work
in flight keeps the engine fed); once it reaches the tighter ceiling, adding
batch stops helping — throughput flattens. All measurements you are given lie
below that flattening point: in that visible range throughput is still
climbing roughly with `x`. The break point ("knee") itself is never shown to
you.

**Illustrative FORM only — NOT the hidden law.** If you were instead fitting,
say, a projectile's height vs. time, you might submit an expression shaped
like `4.9*x**2 - 3*x + 1`. That is only to show the *syntax* your answer must
use; the real relationship here has a different shape you must discover from
the data and the given constants.

## Input (stdin)
```
n t
C W F D
x[0] y[0]
x[1] y[1]
...
x[n-1] y[n-1]
```
`t` is the test id. `C, W, F, D` are the hardware/kernel constants described
above (floats). Then `n` training rows follow: a batch size `x[i]` (integer)
and its measured throughput `y[i]` (float, with measurement noise).

## Output (stdout): one expression

Print **one line** containing a single arithmetic expression over the
variables `x` (batch size), `C`, `W`, `F`, `D`, using `+ - * / **`, unary
`+`/`-`, parentheses, numeric constants, and the functions `sqrt`, `log`,
`exp`, `absv` (one argument each) and `min`, `max` (two arguments each). No
other names, calls, or syntax are allowed. The expression is at most 2000
characters and at most 60 AST nodes.

## Feasibility

The output must parse under the grammar above using only the listed
names/functions with finite numeric constants. When evaluated at every
held-out batch size (substituting that instance's own `C, W, F, D`), it must
produce a finite, strictly positive number. Any violation scores `0`.

## Objective (maximise)

The grader evaluates your expression at several batch sizes far larger than
any you were shown — deep in the regime past the knee — against the true
(lightly noised) throughput there. For each held-out batch `i` with your
prediction `p_i` and true value `v_i`:
```
e_i = min(1.0, |ln(p_i / v_i)|)          # capped log-ratio error
Fq  = 1 / (mean(e_i) + 0.10)
```
`Fq` is compared against `B`, the same `Fq` formula computed for the
checker's own trivial baseline — the constant predictor equal to the mean of
your training throughputs:
```
Ratio = min(1000, 100 * Fq / B) / 1000
```
A predictor that just reproduces the training mean scores `Ratio ~= 0.1`. A
predictor whose growth never bends over will rack up large log-ratio error
once the true curve has flattened and the prediction has not. Getting the
*ceiling* right, and getting the *shape* of the approach to it right, both
move the score — but the held-out throughputs carry their own measurement
noise, so no expression reaches a perfect score.

## Example (worked score, illustrative numbers only)

Suppose two held-out points have true values `v = [100, 100]` and your
expression predicts `p = [100, 100]` (a perfect match): each `e_i = 0`, so
`Fq = 1/0.10 = 10.0`. If the baseline's `Fq` there were `1.0`, this would
give `Ratio = min(1000, 100*10.0/1.0)/1000 = 1.0`. Real instances keep you
well short of that because of noise and because you only see the sub-knee
range of `x`.

## Constraints

Time limit 3 s, memory 512 MB. `n = 10` training rows. Each `.in` file is a
few hundred bytes. Scoring is fully deterministic given the test id.
