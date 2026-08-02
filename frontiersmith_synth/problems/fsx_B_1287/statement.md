# No-Trade Band: Rebalancing Without Bleeding Costs

## Problem

You are hedging a short option position against its underlying over a known,
deterministic price path of `N` timesteps. At each step `t` (`t = 0..N`) the
position's **target hedge ratio** `D[t]` (its delta, in `[-1, 1]`) and its local
**convexity** `G[t] >= 0` (its gamma) are given. You start already delta-neutral:
your held position at `t = 0` is fixed at `h[0] = D[0]`.

At every step `t = 1..N` you choose a held position `h[t]` (the number of
underlying units you hold from `t` onward). Moving from `h[t-1]` to `h[t]`
costs money (a fixed fee plus a per-unit fee), and *not* moving costs money too:
while you hold a stale position through the price move that revealed `D[t]`, you
absorb a **gamma-drift** penalty proportional to how convex the position was and
how far your held amount was from the new target. Crucially, this drift penalty
is charged using `h[t-1]` -- the position you had *before* seeing `D[t]` -- so
trading at time `t` cannot undo it; it can only change what you pay for the
*next* interval. This is why sudden, discontinuous target jumps (and the partial
reversals that sometimes follow them) leave an unavoidable residual cost:
continuous hedging cannot outrun a jump that has already happened.

Your goal: choose `h[1], ..., h[N]` to minimize the total cost.

## Input (stdin)

```
N
S_0 S_1 ... S_N        (N+1 floats: the underlying price path)
D_0 D_1 ... D_N        (N+1 floats: target hedge ratio / delta, each in [-1,1])
G_0 G_1 ... G_N        (N+1 floats: local gamma, each >= 0)
cost_prop cost_fixed   (two floats: proportional and fixed trading costs)
```
`D[0] = 0` and `h[0] = D[0]` always (you start delta-neutral). The price path may
contain occasional large moves, some of which are followed by a partial reversal
on the very next step.

## Output (stdout)

Exactly `N` real numbers `h[1] h[2] ... h[N]` (whitespace/newline separated),
your held position after each step's decision.

## Feasibility

* Exactly `N` finite tokens, each parseable as a real number.
* Every `h[t]` must satisfy `-3.0 <= h[t] <= 3.0`.
* Any violation (wrong count, non-numeric, non-finite, or out of bounds)
  scores `Ratio: 0.0`.

## Objective (minimize)

For `t = 1..N`, with `h[0]` fixed as above:

```
drift[t] = G[t] * (h[t-1] - D[t])^2
trade[t] = cost_fixed * [h[t] != h[t-1]]  +  cost_prop * |h[t] - h[t-1]| * S[t]
F        = sum_t ( drift[t] + trade[t] )
```

## Scoring

The checker also builds its own baseline `B`: the cost of the trivial static
hedge that holds `h[t] = D[0]` for the entire path (never rebalances). Your
score is
```
Ratio = min(1.0, 0.1 * B / F)
```
Lower `F` (relative to `B`) scores higher. The formula's shape is fixed; the
exact cost/gamma numbers live in the input, so a fixed rule of thumb will not
transfer across test cases -- you must read and react to them.

## Constraints

`1 <= N <= 132`, `-1 <= D[t] <= 1`, `G[t] >= 0`,
`cost_prop, cost_fixed > 0`. Time limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)

Suppose `N = 2`, `D = [0, 0.5, 0.4]`, `G = [0, 1.0, 1.0]`,
`cost_prop = 0.001`, `cost_fixed = 0.05`, `S = [100, 101, 99]`. A submission
`h = [0.5, 0.5]` pays `drift[1] = 1.0*(0-0.5)^2 = 0.25`, `trade[1] = 0.05 +
0.001*0.5*101 = 0.1005`, then `drift[2] = 1.0*(0.5-0.4)^2 = 0.01`, `trade[2] = 0`
(no trade at t=2). `F = 0.25 + 0.1005 + 0.01 = 0.3605`. The static baseline
`h=[0,0]` pays `drift[1] = 1.0*(0-0.5)^2=0.25`, `drift[2] = 1.0*(0-0.4)^2=0.16`,
`trade = 0`, so `B = 0.41`. `Ratio = min(1, 0.1*0.41/0.3605) = 0.1138`.
