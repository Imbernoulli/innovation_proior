# Community Diffusion: Minimal-Op Circuit for a Saturating Fixed Point

## Problem
A network of `N` agents is partitioned into `K` communities. Agent `i` belongs
to community `c_i`. Influence between two *individual* agents depends only on
the communities they belong to: every ordered pair of communities `(a, b)`
has a fixed non-negative integer coupling weight `W[a][b]`, applied to *every*
individual edge from an agent of community `b` to an agent of community `a`.
Each community `a` also has a per-community bias `beta[a]` (an integer in
`[1, CAP]`), and there is a global saturation cap `CAP`.

Starting from `x_i^(0) = 0` for every agent, values update synchronously,
round by round:

```
x_i^(t+1) = min( CAP,  beta[c_i] + sum_{j != i} W[c_i][c_j] * x_j^(t) )
```

Because the update only ever depends on an agent's community (never its
individual identity) and the start state `x^(0) = 0` is identical for every
agent in a community, **every agent in the same community follows exactly
the same trajectory forever**. Also, since all weights and biases are
non-negative, the sequence `x^(t)` is coordinate-wise non-decreasing and
bounded above by `CAP`; being integer-valued, it must stabilize at an exact
fixed point `x*` after at most `K * CAP` rounds — a bound that depends only
on `K` and `CAP`, **not** on the actual values of `beta` (see below).

**`beta` is NOT given to you.** Your circuit must be correct for *every*
`beta` the checker chooses from `[1, CAP]^K` — you know `N, K, CAP`, the
coupling table `W`, and the community assignment, but the bias vector is an
unknown input to your circuit, not a number you get to look up. This is
exactly why the `K * CAP` round bound above being beta-independent matters:
it is what lets you unroll "enough" rounds without knowing beta's value.

## Input (stdin)
```
N K CAP
K lines, each: W[a][0] ... W[a][K-1]
c_0 c_1 ... c_{N-1}
```
`1 <= K <= N <= 200`; `1 <= CAP <= 50`; `0 <= W[a][b] <= 3`; `0 <= c_i < K`.

## Output (stdout): a straight-line program (SLP)
```
L
L lines, each one of:
    const v            # a new node holding the integer constant v
    add i j             # node[i] + node[j]
    sub i j             # node[i] - node[j]
    mul i j             # node[i] * node[j]
    min i j             # min(node[i], node[j])
out q_0 q_1 ... q_{N-1}
```
Nodes `0 .. K-1` are **implicit input nodes**: node `a` holds `beta[a]`, the
unknown bias of community `a`. You never emit them — they already exist.
Instruction `t` (0-based) defines node `K + t`; it may reference only
**earlier** nodes (index `< K + t`), so the program is a DAG. The `out` line
names, for every agent `0..N-1`, the node that must equal that agent's
converged value `x*_i` — **for every substituted `beta`**. Two `out` entries
**may point at the same node** — that reuse costs nothing. No tokens may
follow the `out` line's `N` indices.

## Feasibility
Every `add/sub/mul/min` reference must satisfy `0 <= i, j < K+t`; every
`const` value and every intermediate/final node value must stay in
`[-10^15, 10^15]` (reject `nan`/`inf`/non-integer/trailing tokens and any
blow-up); the `out` line must list exactly `N` valid node indices, followed
by nothing else. The checker then substitutes several independently-chosen
bias vectors `beta` (each drawn from `[1, CAP]^K`) into nodes `0..K-1`,
executes your program in exact integer arithmetic for each one, and checks
that node `q_i` equals `x*_i` under *that* `beta`. A single mismatch, under
any of these trials, or any parse/range violation scores `Ratio: 0.0`. (A
circuit that only special-cases one hardcoded `beta` will fail almost every
trial — the game is a genuine circuit, not a memorized number.)

## Objective (minimize) — the cost model
Cost = the number of `add`, `sub`, `mul`, and `min` instructions. `const`
(and referencing the implicit beta nodes) is free — the whole game is
computing as few genuinely new values as possible, in a way that stays
correct across every choice of `beta`.

## Scoring
Let `B` be the op count of the canonical *fully naive* construction: run
`K * CAP` rounds (the round budget any beta-agnostic circuit needs), and in
every round, for every agent, sum every *individual* incoming edge one
multiply-add at a time, then clamp — never sharing work across agents. With
`cost` your submitted circuit's op count,
```
Ratio = min(1.0, 0.1 * B / cost)
```
Reproducing the naive per-agent, per-edge simulation scores about `0.1`.
Communities with many agents but few distinct interaction patterns compress
enormously: since every agent in a community shares one trajectory, the
*entire* dynamics really live in a `K`-dimensional space, not an
`N`-dimensional one — the true minimal circuit is unknown and stays well
below what naive community-level bookkeeping (without collapsing agents)
achieves, so headroom remains.

## Constraints
`1 <= K <= N <= 200`; `CAP <= 50`; `W[a][b] <= 3`; time limit 3 s, memory
512 MB. Scoring is exact-integer and fully deterministic (the checker's
random beta draws are seeded from the instance's own contents).

## Example (illustrative shape only — not from the test set)
Two communities, `N=4` agents (2 per community), `CAP=5`,
`W = [[1,1],[0,0]]`: community 0's agents each see 1 same-community neighbor
and 2 cross-community neighbors (all weight 1); community 1 has no incoming
links at all, so it settles immediately at `beta[1]`. For the *particular*
choice `beta = (1, 1)` the fixed point works out to `x* = (5, 5, 1, 1)` — but
your circuit must get this right for `beta = (1, 1)` *and* every other bias
vector the checker happens to draw from `[1, 5]^2`, e.g. `beta = (3, 4)` gives
a different fixed point. A circuit that computes just the 2 community
representatives (each as a function of the two beta input nodes) and points
all 4 `out` entries at them costs far fewer ops than deriving all 4 agents'
trajectories independently — the connectivity and multiplicities that make
this pay off live in the input's `W` table and community sizes, not in this
example.
