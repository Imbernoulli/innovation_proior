# Island Microgrid Dispatch Under a Mandatory N-1 Spinning-Reserve Rule

## Problem

An isolated island grid is served by `N` generating units and must meet a demand
series over `T` time steps. Every unit `i` has capacity `cap_i` and a strictly
convex fuel-rate curve peaked at a **sweet-spot load** `m_i`:

```
cost_i(p) = a_i * (p - m_i)^2 + b_i * p
```

`a_i > 0` is the curvature (how sharply fuel use rises when the unit is pushed
away from `m_i`) and `b_i` is a linear fuel-rate offset. All units run online at
every time step (no on/off decision); you only choose each unit's output level
`p_{i,t} in [0, cap_i]`.

Some units are flagged **fast** (reserve-eligible); the rest are **slow**. One
designated unit, index `J`, is the largest on the island and is the grid's
**swing unit**: the plant most likely to trip. The grid operator enforces a
hard **N-1 rule**: if `J` trips at time `t`, the *other* fast units' spare
capacity must be enough to instantly cover `J`'s lost output:

```
sum_{i fast, i != J} (cap_i - p_{i,t})  >=  p_{J,t}
```

Slow units get no credit toward this rule even if they have spare capacity.
Note this means reserve is not free: keeping enough headroom on the fast
units to satisfy this rule pulls their loads away from wherever the raw
economics alone would put them.

## Input (stdin)

```
N T
cap_1 m_1 a_1 b_1 fast_1
...
cap_N m_N a_N b_N fast_N
J
D_1 D_2 ... D_T
```
`cap_i` is an integer; `m_i, a_i, b_i` are floats; `fast_i` is `0` or `1`;
`J` (1-indexed) names the swing unit; `D_t` are floats, the demand at each
time step. It is guaranteed `D_t <= cap_total - cap_J`, so a feasible dispatch
always exists (e.g. run `J` at 0 and split `D_t` over the rest by capacity).

## Output (stdout)

`T` lines, each with `N` space-separated numbers: the dispatch `p_{i,t}` for
`i = 1..N` at time step `t = 1..T` (row-major, one time step per line).

## Feasibility

For every `t`: (1) `0 <= p_{i,t} <= cap_i` for all `i`; (2) `sum_i p_{i,t} = D_t`
(demand met exactly); (3) the N-1 rule above holds. Any violation, any
malformed/incomplete/non-finite output, or a wrong token count scores `0`.

## Objective

Minimize total fuel across the whole horizon:
`sum_t sum_i [ a_i*(p_{i,t}-m_i)^2 + b_i*p_{i,t} ]`.

## Scoring

The checker computes your total fuel `F` and an internal baseline `B` (a
simple always-feasible construction: park `J` at 0, split demand over the
rest proportional to capacity, ignoring the efficiency curves). Score is
`min(1000, 100*B/F) / 1000`, printed as `Ratio: <float>`. Lower fuel is
better; a 10x-better solution than the baseline caps the score at `1.0`.

## Constraints

`4 <= N <= 9`, `6 <= T <= 28`, `1 <= cap_i <= 300`, `0 < m_i < cap_i`,
`a_i, b_i > 0`. Time limit 5s, memory 512MB.

## Example (worked, illustrative shape only)

Input:
```
2 1
100 50.0 0.01 1.0 0
80 40.0 0.01 1.0 1
1
70.0
```
Here unit 1 is `J` (index 1, largest), unit 2 is fast. A feasible output for
this one time step is a single line `"30.0 40.0"`: it meets demand
(`30+40=70`) and satisfies the N-1 rule (`80-40=40 >= 30`). This tiny
2-unit example is only for illustrating the input/output SHAPE -- whether a
given feasible split is close to optimal depends on `a_i,b_i,m_i`, which the
checker evaluates exactly via the cost formula above. The real instances
have several fast units with very different curvatures, which is where the
interesting trade-off lives.
