# Clock the Mystery Pipeline

You have a mystery CPU, a stopwatch, and a logbook. The CPU executes short
straight-line programs over four opcodes — `ADD`, `MUL`, `LOAD`, `STORE` —
issuing at most one instruction per cycle, in program order. Two things can
stall it: a data dependency between adjacent instructions (a *register*
written by one instruction and read by the next), which is only partly
hidden by an internal forwarding path, and a *structural* hazard — the
multiply unit is a single, non-pipelined resource, so back-to-back `MUL`s
contend for it.

Rather than raw source code, your logbook records eight structural counts
per probe program plus its stopwatch-measured cycle count:

| column | meaning |
|---|---|
| `n`   | instruction count |
| `nA nM nL nS` | opcode counts (`ADD MUL LOAD STORE`); they sum to `n` |
| `cLU` | # instructions immediately after a `LOAD` that data-depend on it |
| `cMF` | # instructions immediately after a `MUL` that data-depend on it |
| `cST` | # adjacent `MUL,MUL` pairs (the structural-hazard count) |

Every probe program has **at most one contiguous run of consecutive `MUL`s**
(all other `MUL`s, if any, stand alone); `cST` is exactly that run's length
minus one (`0` if there is no run of two or more).

## Input (stdin)
```
T t
n1 nA1 nM1 nL1 nS1 cLU1 cMF1 cST1 cycles1
...
nT ... cyclesT
```
`t` is the case id; `T` training probes follow. **The training probes are
all short and their MUL runs are always length 0-3** (so `cST` never exceeds
2 here). The grading probes are a different, disjoint set — longer programs
with much longer MUL runs — regenerated only by the grader; you never see
them.

## Output (stdout)
One line: a closed-form Python expression for the cycle count, in the
variables `n nA nM nL nS cLU cMF cST`. Allowed: `+ - * / **`, unary `-`,
numeric constants, and the functions `sqrt log exp sig tanh absv`. Example
(illustrative **form only — NOT the hidden law**): `n + sqrt(nM) - 0.2*nS`.
No other names are accepted; at most 80 expression nodes.

## Feasibility
The expression must parse under the grammar above and evaluate to a finite
real number on every grading probe. Any violation scores `0`.

## Objective (maximise)
Let `p_i` be your prediction and `t_i` the true (stopwatch-noisy) cycle
count on grading probe `i`. The grader forms
```
metric   = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
O        = metric * (1 + LAMBDA * nodes)          # nodes = expression size
baseline = the same metric for the constant predictor mean(train cycles)
Ratio    = min(1000, 100 * baseline / O) / 1000
```
Lower held-out error raises `Ratio` (capped at `1.0`); `LAMBDA` is a small
parsimony weight. A constant predictor scores about `0.1`. Stopwatch jitter
is irreducible, so even the true law does not reach `1.0` — there is room
above the reference solutions.

## Why the obvious fit is a trap
On the training probes, `cST` never exceeds 2, and over that tiny range a
plain linear term `b*cST` already tracks the data closely — a per-hazard
additive regression (one constant cost per adjacent `MUL` pair) fits the
training table almost exactly. But the multiply unit is a single server: a
run of `R` back-to-back `MUL`s each waiting on it backlogs like a queue
whose service time exceeds its arrival rate, and that backlog grows with
the **triangular number** of the run length, not linearly with the number
of adjacent pairs. A model calibrated only on runs of length 0-3 cannot
tell a linear cost from a quadratic one — the two curves nearly coincide
there — and it is graded on runs many times longer, where they diverge
enormously.

## Worked example (mechanics only, not the real law)
Suppose (illustrative only) the true law were `cycles = n + cLU`, and one
grading probe has `n=10, cLU=3` so `t=13`. Submitting `n + cLU` predicts
`p=13`, giving a per-point bounded error of `0`. Submitting the constant
`10` gives error `3/23 ≈ 0.130`. Averaged over all grading probes and run
through the `O`/`baseline` ratio above, the exact-law submission scores
near the ceiling and the constant reproduces the `~0.1` baseline — this is
only to show the arithmetic, not the actual hidden law.

## Constraints
Time limit 5 s, memory 512 MB. `T` is at most a few hundred rows. Scoring is
fully deterministic.
