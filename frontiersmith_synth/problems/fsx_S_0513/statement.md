# Roadwork: Cutting Streets to Un-jam a Selfish City

A toll-free city routes a fixed volume of self-interested drivers from a source `s`
to a sink `t`. Every driver picks the fastest route given what everyone else does,
so traffic settles into a **Wardrop (selfish-routing) equilibrium**. Counter-intuitively,
some streets make the equilibrium *worse*: closing them can strictly lower everyone's
travel time (Braess's paradox). Your job is to choose **which streets to keep open**.

## Input (stdin)

```
N M s t k D
u_1 v_1 a_1 b_1
...
u_M v_M a_M b_M
```

- `N` nodes (`0..N-1`), `M` directed edges, source `s`, sink `t`.
- `k` (integer, `k >= 2`) is the congestion exponent; `D` (float) is the total driver demand.
- Edge `i` (1-based) goes `u_i -> v_i` with latency
  `ell_i(x) = a_i + b_i * x^k`, where `x` is the flow on that edge.
  All `b_i > 0`, so latencies strictly increase and the equilibrium flow is unique.

## Output (stdout)

A whitespace-separated list of **1-based edge indices to KEEP**. The remaining edges are
closed. Order does not matter; duplicates are ignored. (Print nothing extra.)

## Feasibility

The kept edges must still allow all demand to travel from `s` to `t` — i.e. `t` must be
reachable from `s` using only kept edges. Any index outside `1..M`, any non-integer token,
or a kept subnetwork that disconnects `s` from `t` is **infeasible** and scores `0`.

## Objective (minimize)

On the *kept* subnetwork, the checker computes the unique Wardrop equilibrium: a feasible
`s->t` flow of value `D` in which every used path has minimum latency (no driver can switch
to a cheaper route). It does this by deterministic Frank-Wolfe convex minimization of the
Beckmann potential `sum_i integral_0^{x_i} ell_i(z) dz` to a fixed iteration budget. Your
cost is the resulting **total travel time**

```
TT = sum over kept edges i of  x_i * ell_i(x_i) = sum_i ( a_i*x_i + b_i*x_i^(k+1) ).
```

Smaller is better. Because equilibrium travel time is **non-monotone in the edge set**,
`TT` is *not* minimized by keeping every road: a tempting low-latency shortcut can pull
flow onto already-congested edges and raise `TT`, while a different-looking extra edge may
add genuine capacity and lower it. You cannot tell which is which by inspecting the graph —
you must reason about the equilibrium the kept edges induce.

## Scoring

Let `F` be your kept subnetwork's equilibrium total travel time and let `B` be the checker's
baseline: route all `D` drivers down the single free-flow shortest path (a congestion-blind
router). The score is

```
Ratio = min(1.0, 0.1 * B / F).
```

Keeping the whole network already beats the single-path baseline (`Ratio > 0.1`); pruning the
harmful shortcuts beats that again. The `0.1` scale leaves ample headroom above any reference
solution. Each of the 10 test cases is scored independently and the mean `Ratio` is reported.

## Constraints

- `N <= 40`, `M <= 40`, `2 <= k <= 4`, `7 <= D <= 16`. Time limit 5 s, memory 512 MB.
- Scoring is fully deterministic: same output ⇒ same score, on any machine.

## Example (illustrative)

For the classic Braess diamond `s->a` (fast-but-congestible), `a->t` (slow-but-fixed),
`s->b` (slow-but-fixed), `b->t` (fast-but-congestible), plus a near-zero shortcut `a->b`:
keeping all five edges funnels every driver through both congestible edges (`s->a->b->t`),
whereas **deleting the `a->b` shortcut** splits traffic across the two balanced routes and
lowers the equilibrium travel time. The optimal set of closures on a multi-layer network is
not given by any single greedy rule.
