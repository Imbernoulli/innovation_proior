# Interference-Free Deep-Sea Cable Backbone

## Problem

A deep-sea cable operator plans a backbone over a family of candidate **routes**.
Each candidate route is identified by a length-`n` string of trits (digits in
`{0,1,2}`) — its *relay signature* — so the routes are exactly the `3^n` vectors
of the affine space `AG(n,3) = F_3^n`. Route `v` carries an integer **throughput**
`w(v) >= 1`.

Three **distinct** routes `a, b, c` suffer a destructive resonance (they *interfere*)
exactly when

```
a + b + c == 0  (mod 3),  coordinate by coordinate,
```

i.e. when their signatures are collinear in `AG(n,3)`. A deployable backbone is a
set of routes containing **no** interfering triple — a *cap set*.

Deploy an interference-free backbone of **maximum total throughput**.

## Input (stdin)

```
n
w_0 w_1 ... w_{3^n - 1}
```

- Line 1: the dimension `n`.
- Line 2: `3^n` integers, the throughput of every route in **canonical base-3
  order** — route with index `i` has trits equal to the base-3 digits of `i`
  with the most significant coordinate first (e.g. for `n = 3`, index `5` is
  `012`).

## Output (stdout)

```
k
r_1
r_2
...
r_k
```

- Line 1: the number `k` of deployed routes.
- Each following line: one route as a length-`n` trit string over `{0,1,2}`.
- Routes must be distinct.

## Feasibility

The output is feasible iff every listed route is a valid length-`n` trit string,
all routes are distinct, and the set contains **no** interfering (collinear)
triple `a + b + c == 0 (mod 3)`. Any violation scores `0`.

## Objective

Maximize `F = sum of w(v)` over the deployed routes `v`.

## Scoring

Let `B` be the throughput of a trivial reference backbone that the grader builds
itself: a **weight-blind** greedy pass over routes in canonical order (which
simply deploys the 0/1 sub-cube). With your feasible throughput `F`,

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

So the trivial reference scores `Ratio ≈ 0.1`, and a backbone that beats it by
`10x` would cap at `1.0`. The maximum-weight cap set is not known in closed form,
so there is genuine headroom above any heuristic.

## Constraints

- `n` ranges from `4` to `7` across the test ladder (`3^n` up to `2187` routes).
- Weights are integers in `[1, 1000]`.
- Scoring is exact integer arithmetic and fully deterministic.

## Example (worked score, illustrative — small `n`)

For `n = 4` (`81` routes), the weight-blind reference deploys the 16 routes of
the 0/1 sub-cube for some throughput `B`. A weight-steered greedy that processes
routes in descending throughput order typically deploys a different 16–20 route
cap of throughput `F ≈ 2.5 B`, scoring `Ratio ≈ 0.25`. A better priority (or a
local exchange search) can push higher — the optimum is open.
