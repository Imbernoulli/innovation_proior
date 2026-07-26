# Batched Calibration-Curve Evaluation: One Shared Circuit, Three Query Shapes

## Problem

A sensor rig holds one fixed degree-`n` integer calibration polynomial
`P(x) = a[0] + a[1] x + ... + a[n] x^n`. Every duty cycle it must report `P(x)`
at every point of THREE query batches that arrive together:

* **sweep** -- an arithmetic-progression scan `x0, x0+h, x0+2h, ...` (many points);
* **probe** -- repeated calibration probes: only a HANDFUL of distinct relative
  offsets around a moving reference point, each probed several times, in
  scrambled order;
* **ad-hoc** -- scattered one-off diagnostic queries with no shared structure.

On this rig, a scalar **multiplication** is the scarce resource (each burns a
hardware MAC cycle); additions/subtractions are effectively free. You must
emit ONE shared **straight-line program (SLP)** that correctly reports every
query across all three batches, using as few `mul` instructions as possible.
Evaluating every point from scratch with Horner's rule is never optimal here:
the batches are handed to you with exploitable shape, and the real win is
choosing a *shared preprocessing prefix* that more than one batch's shortcut
can build on, instead of re-deriving every one of the `M` query points
independently.

## Input (stdin)

```
testId
n
a[0] a[1] ... a[n]            (n+1 integers, a[n] != 0)
Q
q[0] q[1] ... q[Q-1]          (Q distinct integers -- the query value pool)
m1
idx1[0] ... idx1[m1-1]        (sweep batch: indices into q[])
m2
idx2[0] ... idx2[m2-1]        (probe batch: indices into q[])
m3
idx3[0] ... idx3[m3-1]        (ad-hoc batch: indices into q[])
```

## Output (stdout) -- a shared straight-line program

```
L
L lines: <op> <arg1> <arg2>          op in {mul, add, sub}
M                                     (M = m1+m2+m3)
M lines: <register index>            outputs, in order: sweep, probe, ad-hoc
```

Each `arg` is one of `a<k>` (0<=k<=n, the given coefficients), `q<j>`
(0<=j<Q, the given query values), or `r<i>` (the result of instruction `i`,
`0 <= i <` the current instruction's own index -- backward references only).
There are no free numeric literals: every leaf must be a value the instance
actually gave you, so a program cannot simply print the answer.

## Feasibility

Every instruction and every output line must be well-formed and in range.
The program, when executed, must produce EXACTLY the right value (integer
equality) at every one of the `M` output positions -- checked not only
against the instance you were given, but also against two independently
regenerated instances that share the identical batch structure (same `n`,
`m1/m2/m3`, same probe repeat pattern) with fresh random numbers. A program
whose correctness is a numeric coincidence for one specific set of numbers
will not also satisfy two unrelated random instances; any violation scores
`Ratio: 0.0`.

## Objective (minimize)

`F` = the number of `mul` instructions in your program.

## Scoring

Let `B = (2n-1) * M`, the cost of the naive per-point approach (build the
power ladder `x^2..x^n` from scratch for every query, then dot with the
coefficients). The checker verifies exact equivalence, then reports

```
Ratio = min(1, 0.1 * B / F).
```

Matching that naive cost scores `0.1`; a program using `10x` fewer
multiplications caps the score at `1.0`. The minimum achievable `F` for a
jointly-optimal three-batch schedule is not something this checker computes,
so headroom always remains above any reference solution shipped here.

## Constraints

* `8 <= n <= 11`, `a[n] != 0`, all arithmetic exact integers.
* `1 <= L <= 20000` instructions.
* Deterministic integer arithmetic only -- no floating point.

## Example (worked score)

Suppose `n = 8` and `M = 50`, so `B = 15*50 = 750`. A program using `F = 150`
multiplications scores `Ratio = min(1, 0.1*750/150) = 0.5`. (Illustrative
only -- not a specific instance below.)
