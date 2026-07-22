# Synergy Bin Packing: Dual-Priced Assignment with Hidden Co-location Bonuses

You run a warehouse of `M` capacity-limited bins. There are `N` items; item `i` has weight
`w[i]` and a standalone value `v[i]`. You may put each item in one bin or leave it out.
Certain **pairs** of items reinforce each other: a public **synergy table** lists triples
`(i, j, s)` meaning that if items `i` and `j` end up in the **same** bin you earn an extra
bonus `s` on top of both standalone values. Value is therefore **non-separable** — what an
item is worth depends on what shares its bin.

Write a program that reads one instance and outputs an assignment maximizing total value.

## Input (one JSON object on stdin)

```
{"name": str,
 "N": int, "M": int,
 "C":   [c_0 ... c_{M-1}],     # capacity of each bin (> 0)
 "w":   [w_0 ... w_{N-1}],     # item weights (> 0)
 "v":   [v_0 ... v_{N-1}],     # item standalone values (>= 0)
 "syn": [[i, j, s], ...]}      # co-location bonuses, i < j, s > 0
```

## Output (one JSON object on stdout)

```
{"assign": [b_0 ... b_{N-1}]}   # b_i in {-1, 0, ..., M-1};  -1 = item i left out
```

## Feasibility

`assign` must be a list of exactly `N` integers, each `-1` or a bin index in `[0, M)`. For
every bin `b`, the total weight of items assigned to `b` must not exceed `C[b]`. Any
capacity violation, out-of-range bin, wrong length, non-integer / NaN / boolean entry, a
crash, or a timeout makes the whole instance score **0**.

## Objective (maximize)

For a feasible assignment `a`:

```
obj(a) = sum over i with a[i] != -1 of  v[i]
       + sum over (i, j, s) in syn with a[i] == a[j] != -1 of  s
```

You collect an item's value when it is placed anywhere, and a pair's bonus **only** when
both its members share one bin.

## Scoring (deterministic)

Let `U = sum(v) + sum(s over syn)` — the (unreachable) value of admitting every item and
co-locating every pair. Your per-instance score is

```
r = clamp( 0.1 + 0.9 * obj(a) / U , 0, 1 )
```

Leaving every item out scores exactly `0.1`. Because total weight far exceeds total
capacity, `U` can never be reached, so score headroom always remains above any solution.
The final score is the **mean of `r` over 10 fixed hidden instances** (a mix of
synergy-trap, mixed, and dense-chain layouts). Same submission ⇒ same score.

## Why it is not just a knapsack

The obvious move — sort items by `v/w` and first-fit them into bins — is blind to `syn`.
Many instances hide most of the value inside synergy pairs whose *individual* value/weight
ratios are mediocre, while scattering high-ratio decoy singletons that soak up capacity. A
ratio-greedy pass grabs the decoys, fragments the bins, and rarely co-locates a pair, so it
leaves most of the bonus mass on the table. Treating a worthwhile pair as one merged
super-item, pricing the scarce bin capacity to decide which pairs earn their room, and
repairing the packing when a super-item does not fit recovers value the greedy pass cannot
see. The bonus coefficients live in the input — read and exploit them.
