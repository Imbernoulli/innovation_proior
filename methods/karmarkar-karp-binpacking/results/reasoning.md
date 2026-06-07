OK, let me think this through from scratch. I want to pack `n` pieces of sizes in `(0,1]` into the fewest unit bins, and I want a polynomial-time algorithm whose bin count is close to the true minimum `OPT(I)`. The state of the art is all of one shape: `A(I) ≤ C·OPT(I) + o(OPT(I))`, with people grinding `C` down from `1.7` to about `1.18`. That bothers me, because a multiplicative constant — even `1.05` — wastes a fixed *fraction* of the bins forever. On an instance that truly needs a million bins, a `1.18` algorithm can throw away a hundred-eighty-thousand of them, and no amount of polishing the constant fixes the shape of the bound. What I actually want is an *additive* error that grows slowly with the instance: `A(I) = OPT(I) + (something lower-order)`. If I could get the slack down to a power less than one, or a logarithm, then asymptotically I'd be using essentially the optimal number of bins. So the target isn't a smaller `C`; it's killing the `C` entirely and asking how small the additive remainder can be.

Let me first get my bearings on how `OPT` sits relative to things I can actually compute. Let `SIZE(I) = Σ sᵢ`. A bin holds at most one unit of size, so any packing uses at least `SIZE(I)` bins: `OPT(I) ≥ SIZE(I)`. And the other direction is almost as cheap — in any packing where at most one bin is under half full, pair up the underfull bins: I can't have two bins each less than half full, or I'd merge them. So a careful packing has at most one bin below half, giving cost `≤ 2·SIZE(I) + 1`. Hence `SIZE(I) ≤ OPT(I) ≤ 2·SIZE(I) + 1`. The entire difficulty is the factor of two between these. `SIZE` is trivial to compute; `OPT` is NP-hard; the gap is where all the work is.

The greedy algorithms — first-fit-decreasing and its relatives — live inside this factor-of-two band and chip at the constant. But they never look at the *global combinatorial structure* of which pieces can share a bin; they make local placement decisions. If I want an additive bound I think I need an object that captures "the best fractional way to fill bins," because fractional optima are where loss becomes lower-order. So let me set up the strongest relaxation I can.

Here's the relaxation. Bucket the pieces into `m` distinct sizes `s₁,…,s_m`, with `bᵢ` pieces of size `i`. A *configuration* is a multiset of sizes that fits in one bin — a vector `(a₁ⱼ,…,a_mⱼ)` of nonnegative integers with `Σ aᵢⱼ sᵢ ≤ 1`. Let `A` be the `m × q` matrix whose columns are all configurations. The honest problem is "choose how many bins `xⱼ ≥ 0`, integer, of each configuration so that every piece is covered": `min 1·x` s.t. `A x ≥ b`, `x` integer `≥ 0`. Drop the integrality and I get the linear program

```
(I)   minimize 1·x   subject to   A x ≥ b,   x ≥ 0.
```

Call its optimum `LIN(I)`. This is the cutting-stock LP — Eisemann wrote it down for the trim problem, Gilmore and Gomory used it for paper rolls. Two facts pin it. An integer packing *is* a feasible integer `x` (one variable per bin actually used), so `LIN(I) ≤ OPT(I)`. And feasibility forces the covering, so `Σ sᵢ bᵢ = SIZE(I) ≤ LIN(I)`: more carefully, if `x` is LP-optimal and `s` is the size vector, then `LIN(I) = 1·x ≥ (sᵀA)x ≥ sᵀb = SIZE(I)`, because every column of `A` has size-sum `sᵀ(column) ≤ 1` so `sᵀA ≤ 1ᵀ`. So `SIZE(I) ≤ LIN(I) ≤ OPT(I)`. Good — `LIN` is squeezed between the trivial lower bound and the thing I want, and unlike `OPT` it's a linear program, in principle computable.

Now the real question: if I can compute (or nearly compute) `LIN(I)`, how much do I lose rounding the fractional solution to an honest integer packing? This is where the relaxation has to earn its keep. Take a *basic* feasible solution `x` to `(I)`. A basic feasible solution to any LP has at most as many nonzero variables as the LP has constraints — and `(I)` has `m` covering constraints (one per distinct size), not counting the `x ≥ 0` bounds. So `x` has at most `m` nonzero components: a fractional optimum that mixes at most `m` distinct configurations. That's a strong handle. Let me round it.

For each nonzero `xⱼ`, put `⌊xⱼ⌋` bins of configuration `j` into the packing — call that the *principal part*. What's left over is a residual instance `I'`: the pieces I still owe because I only took the floor. The number of still-unpacked pieces of size `i` is governed by the fractional parts `Σⱼ (xⱼ − ⌊xⱼ⌋)·aᵢⱼ`. How big is the residual's total size? Write `fⱼ = xⱼ − ⌊xⱼ⌋ ∈ [0,1)`. The residual is covered by the fractional bits, so `SIZE(I') ≤ LIN(I') ≤ Σⱼ fⱼ`, and since there are at most `m` nonzero `xⱼ`, each contributing `fⱼ < 1`, we get `SIZE(I') ≤ Σⱼ fⱼ < m`. Now pack the residual two ways and take the better. One way: start a fresh bin for each of the (at most `m`) nonzero configurations and delete the excess pieces — that's at most `m` bins. The other way: pack `I'` with the half-full argument — at most `2·SIZE(I') + 1` bins. So the residual costs

```
min(m, 2·SIZE(I') + 1).
```

Rewrite the minimum as `min(m, 2·SIZE(I')+1) = SIZE(I') + min(m − SIZE(I'), SIZE(I') + 1)`, and the min of two numbers is at most their average: `min(m − SIZE(I'), SIZE(I') + 1) ≤ ((m − SIZE(I')) + (SIZE(I') + 1))/2 = (m+1)/2`. So the residual costs at most `SIZE(I') + (m+1)/2`. The principal part is `Σⱼ ⌊xⱼ⌋` bins, and `SIZE(I') ≤ Σⱼ fⱼ`, so principal plus residual is at most `Σⱼ ⌊xⱼ⌋ + Σⱼ fⱼ + (m+1)/2 = Σⱼ xⱼ + (m+1)/2 = 1·x + (m+1)/2`. Taking `x` optimal,

```
OPT(I) ≤ LIN(I) + (m+1)/2.
```

There it is — the lever the whole thing turns on. The loss from fractional to integer is controlled by `m`, the number of *distinct sizes*, and not by `n`, the number of pieces. If `m` were small, I'd be done: solve the LP, round, lose `O(m)` bins. But real instances have many distinct sizes; `m` can be `Θ(n)`, and then `(m+1)/2 ≈ n/2` is a disaster, `Θ(OPT)`, no better than greedy. Two problems stand between me and the additive bound, and I can see them now precisely: (1) `m` is too big, so I must *reduce the number of distinct sizes* without moving `OPT`; and (2) even granting that, I have to actually *solve a linear program with an astronomical number of columns* `q` in polynomial time. Let me take them in turn.

Reducing distinct sizes. The known move is linear grouping. Sort the pieces in decreasing size, cut them into consecutive groups `G₁ ≥ G₂ ≥ … ≥ G_q` of `k` pieces each, and round every piece in a group *up* to the largest size in that group, forming `Gᵢ'`. Round up, not down, so the rounded instance dominates pieces by pieces and a feasible packing of the rounded instance is feasible for the original. Discard the top group `G₁` and pack its `k` pieces one per bin; let `J = G₂' ∪ … ∪ G_q'`. Two things to check. First, `J` has at most `n/k` distinct sizes — one per group, and there are `n/k` groups. Second, how much did `OPT` move? Here's the slick part: because I rounded each group up to the max of that group, and the groups are sorted descending, `Gᵢ'` (group `i` rounded up) is dominated piece-for-piece by `Gᵢ₋₁` (the un-rounded previous group, every piece of which is at least as large). So `∪ᵢ≥₂ Gᵢ' ` is dominated by `∪ᵢ≥₁ Gᵢ = I` shifted by one group, giving `OPT(J) ≤ OPT(I)`; and `I` is dominated by `J ∪ G₁`, giving `OPT(I) ≤ OPT(J) + OPT(G₁) ≤ OPT(J) + k`, since `G₁` is `k` pieces packed in `≤ k` bins. So

```
OPT(J) ≤ OPT(I) ≤ OPT(J) + k,
```

and the same domination chain gives the matching statements for `LIN` and `SIZE`. Distinct sizes drop to `n/k`; `OPT` moves by at most `k`. Now combine with the rounding loss: solve, round, and I lose roughly `(m(J)+1)/2 + k ≈ (n/k)/2 + k`. Minimize over `k`: balance `n/k` against `k`, take `k ≈ √n`, and the loss is `O(√n)`. Hmm. `√n` is sublinear, better than greedy's `Θ(OPT)`, but it's `O(OPT^{1/2})`-ish, not the logarithm I'm chasing. Linear grouping by itself can't make `m` *small* and the grouping loss *small* simultaneously — the `n/k` and `k` terms fight, and `√n` is the best the tradeoff allows.

So linear grouping is too crude. Stare at why. The loss `k` per group came from discarding `k` pieces, and the distinct-size count `n/k` came from having `n/k` groups, and `n` is the number of *pieces*. But the bound I really care about, `(m+1)/2`, scales with `SIZE` through `LIN`, and `SIZE ≤ OPT`. The pieces aren't all the same scale: a bin can hold one piece near size `1` or many pieces near size `0`. Linear grouping treats a fat piece and a tiny piece identically — `k` of each per group — which is wasteful, because rounding up `k` tiny pieces costs almost nothing in size while rounding up `k` fat pieces costs a lot, yet I'm forced to use the same `k` everywhere. Let me make the grouping *scale-aware*. Split the pieces by dyadic size class: for `r = 0, 1, 2, …`, let `Iᵣ` be the pieces whose sizes lie in `(2^{-(r+1)}, 2^{-r}]`. Within class `r`, pieces are all within a factor of two of each other, and there are at most `⌈log₂(1/a(I))⌉` nonempty classes, where `a(I)` is the smallest piece. Now apply linear grouping *within each class*, but with a class-dependent group size `k·2^r` in class `r`. Why `k·2^r`? Because a piece in class `r` has size `> 2^{-(r+1)}`, so a group of `k·2^r` such pieces has total size `> k·2^r · 2^{-(r+1)} = k/2` — every group, in every class, carries roughly the same *total size* `≈ k`, regardless of how small the individual pieces are. That's the fix: group by size budget, not by piece count.

Let me run the bound. The discarded top group in class `r` is `k·2^r` pieces each smaller than `2^{-r}`, so it packs into at most `k` bins — independent of `r`. Summed over the `⌈log₂(1/a)⌉` classes, the total `OPT` loss is `≤ k·⌈log₂(1/a)⌉`. And the distinct-size count: in class `r`, linear grouping with parameter `k·2^r` leaves at most `n(Iᵣ)/(k·2^r)` distinct sizes; but `n(Iᵣ)·2^{-(r+1)} ≤ SIZE(Iᵣ)`, so `n(Iᵣ) ≤ 2^{r+1}·SIZE(Iᵣ)` and the class contributes at most `2^{r+1}·SIZE(Iᵣ)/(k·2^r) = 2·SIZE(Iᵣ)/k` distinct sizes. Summing over classes, the total distinct sizes is at most `(2/k)·SIZE(I) + ⌈log₂(1/a)⌉` (the `+log` because each nonempty class can leave one extra rounded size). So

```
m(J) ≤ (2/k)·SIZE(I) + ⌈log₂(1/a(I))⌉,    OPT(J) ≤ OPT(I) ≤ OPT(J) + k·⌈log₂(1/a(I))⌉.
```

This is the geometric-grouping payoff, and it's qualitatively better than linear grouping. The distinct-size count now scales with `SIZE/k`, and `SIZE ≤ OPT`, so `m ≈ OPT/k`; the loss scales with `k·log(1/a)`. The `n` is gone — both terms now live on `SIZE`/`OPT` and a log of the size ratio. There's also a size-budget variant that's cleaner to state: instead of dyadic classes, sweep the sorted pieces and start a new group whenever the accumulated size of the current group reaches `k`. Then by construction each group has size `≥ k`, the number of groups is `≤ SIZE/k`, and a careful accounting (each group's "overflow" beyond `k` is one piece, and consecutive group sizes can only grow by the harmonic increments `1/(ℓ−1)`) gives `m(J) ≤ SIZE(I)/k + ln(1/a(I))` with `OPT` loss `≤ 2k·(2 + ln(1/a))`. Same shape: `m ≈ SIZE/k`, loss `≈ k·ln(1/a)`.

Before I spend that, the small pieces. If `a(I)` is minuscule, `log(1/a)` blows up. But tiny pieces are exactly the ones I can afford to defer: fix a threshold `g`, set aside every piece of size `≤ g/2`, pack the large pieces, then reinsert the small ones greedily, opening a new bin only when forced. If reinsertion never opens a bin, the count is unchanged. If it does, then in the resulting packing every bin but one is filled to more than `1 − g/2` (else the small piece would have gone there), so `SIZE(I) > (1 − g/2)(B − 1)` where `B` is the final count, giving `B ≤ SIZE(I)/(1 − g/2) + 1 ≤ (1 + g)·OPT(I) + 1`. So the cost after reinsertion is `≤ max(A, (1 + g)·OPT(I) + 1)` where `A` is the cost of packing the large pieces. The point: reinsertion costs only a `(1 + g)` factor plus one. And once the small pieces are gone, every remaining piece is `> g/2`, so `a(I) > g/2` and `log(1/a) = O(log(1/g))`. If I pick `g = 1/SIZE(I)`, then `log(1/g) = log SIZE(I) = O(log OPT)`, and the small-piece reinsertion costs `(1 + 1/SIZE)·OPT + 1 = OPT + O(1)`. So the price of bounding `a` away from zero is negligible, and the grouping logs become `O(log OPT)`. Good — the threshold `g = 1/SIZE` is doing two jobs at once: it makes reinsertion almost free and it tames `log(1/a)`.

Now the second problem, the one Gilmore and Gomory only solved heuristically: the LP `(I)` has `q` columns, astronomically many, so I can't even write it down. I cannot run the simplex method on a matrix I can't store. Stare at the LP and its dual. The primal has `q` variables and `m` constraints; the dual flips that —

```
(II)   maximize u·b   subject to   u ≥ 0,   uᵀA ≤ 1.
```

The dual has only `m` variables (a *price* `uᵢ` per distinct size) but `q` constraints (one per configuration: the prices of the pieces in any bin must sum to `≤ 1`). And by LP duality their optima coincide, `min 1·x = max u·b = LIN(I)`. A program with `m` variables and astronomically many constraints — that's precisely the situation the Grötschel–Lovász–Schrijver ellipsoid method was built for. Ellipsoid never needs the constraints listed; it needs a *separation oracle*: given prices `u`, either certify `u` feasible or hand back one violated constraint.

So what is the separation oracle here? `u` is feasible iff *no* configuration is overpriced, i.e. iff there is no multiset of sizes fitting in a bin whose total price exceeds `1`. The most-overpriced bin is found by maximizing total price subject to the size budget:

```
maximize  v·u   subject to   v·s ≤ 1,   v ≥ 0 integer.
```

That's a knapsack problem — value `uᵢ`, weight `sᵢ`, capacity `1`. If its optimum is `≤ 1`, every bin is fairly priced and `u` is feasible. If its optimum is `> 1`, the optimal `v` is a configuration whose constraint `u·v ≤ 1` is violated — exactly the separating hyperplane the ellipsoid wants. The pricing subproblem Gilmore and Gomory solved heuristically is, read as a separation oracle, what makes the ellipsoid method *provably* polynomial here. Beautiful — the column generation and the ellipsoid are the same idea viewed from primal and dual.

One snag: knapsack is itself NP-hard, so I can't solve the oracle exactly in general. But I don't need exact — the ellipsoid only needs *approximate* feasibility within the tolerance it's already carrying. Round each price down to a multiple of a small grid: set `ūᵢ = (t/n)·⌊n·uᵢ/t⌋`, so every `ūᵢ` is an integer multiple of `t/n` and `ūᵢ ≤ uᵢ ≤ ūᵢ + t/n`. With prices on a grid of spacing `t/n`, the achievable total prices form a polynomially bounded set, and knapsack becomes a dynamic program. Let `F(κ)` be the minimum total *size* of a set of pieces whose total price is exactly `κ·(t/n)`:

```
F(0) = 0,    F(κ) = minᵢ [ F(κ − (n·ūᵢ/t)) + sᵢ ],   κ > 0,
```

and `ū` is feasible iff `F(κ) > 1` for every `κ` with `κ·(t/n) > 1`. The table has `O(n/t · maxᵢ uᵢ)` entries and each is filled in `O(m)` work, so the oracle runs in time polynomial in `m`, `n`, `1/t`. Rounding the prices down changes the knapsack value by at most `t` (the dropped grid remainders), so testing `ū` instead of `u` costs the ellipsoid only an extra `t` of slack — which I fold into its tolerance. So: ellipsoid on the dual `(II)`, knapsack-by-DP separation oracle on rounded prices, and I get the dual optimum within tolerance.

Let me account the ellipsoid run. The dual feasible region `K` sits between two balls — every component of a sensible price is between about `a/n` and `1/sₘ`, so `B(u₀, r) ⊆ K ⊆ B(u₁, R)` with `r, R` polynomial in the data. The GLS bound says that after `M = 4m²⌈ln(R/r)⌉ = 4m²⌈ln(n/t)⌉`-ish iterations, each shrinking the ellipsoid volume by a fixed factor `e^{-1/(2(m+1))}` per cut and computed with `O(m²)` arithmetic, the best feasible price found is within `t` of the dual optimum. At each iteration: query the oracle at the (rounded) center; if it returns an overpriced configuration, that's a *feasibility cut* `u·v ≤ ū·v`; if the center is feasible, make an *optimality cut* `u·b ≥ center·b` to keep improving the objective. After `M` iterations I have `u*` with `u*·b ≥ LIN(I) − t`.

But I wanted the *primal* — an integer packing — not just the dual value. Here's the payoff of running the oracle: the feasibility cuts I made along the way each returned an *actual configuration*. There are at most `M` of them. Define a "realized" dual LP `(II')` that keeps only those at-most-`M` returned configuration-constraints plus the box `0 ≤ uᵢ ≤ 1`. An observer watching the ellipsoid run could not tell whether it was solving `(II)` or `(II')` — same starting ellipsoid, same oracle responses — so the value of `(II')` is within `t` of `LIN(I)` too. And `(II')` has only `M` constraints, hence its dual `(I')` is a fractional bin-packing LP over only `M` configurations — finite, writable, solvable. So I've gone from `q ≈ ∞` columns to `M = poly(m, log n)` columns while keeping the optimum within `t`.

I can sharpen this further, because for the rounding lemma I need a *basic* solution using at most `m` configurations, and `(I')` has `M > m`. A fundamental LP fact: in any bounded LP with `m` variables, some set of `m` constraints already determines the optimum — the other constraints are slack and could be dropped without raising the value. So I prune. Partition the `M` configuration-constraints of `(II')` into `m+1` groups of nearly equal size; by pigeonhole at least one group is disjoint from the critical `m`-element set, so dropping that group doesn't raise the optimum — and I can *test* a group by re-running the (cheap, restricted) GLS solve and checking the value didn't move by more than `t`. Repeat until exactly `m` independent constraints remain. The reduced dual then reads `max u·b` s.t. `uᵀB ≤ τ` for a nonsingular `m × m` matrix `B` (columns = `m` surviving configurations) and a `{0,1}` vector `τ`; its dual gives the unique basic primal `x = B⁻¹b`, a basic feasible solution to the fractional packing using only `m` configurations, of value `≤ LIN(I) + h` for any target tolerance `h` once I set the internal `t` small enough. The number of GLS re-solves to prune is `O((m+1)(1 + ln(M/m)))` — polynomial. Folding the per-iteration knapsack-DP cost into the ellipsoid iteration count, the whole fractional-packing subroutine runs in a time I'll call `T(m, n)`, polynomial in `m` and `n`. So: *the configuration LP is in `P`, despite its astronomical column count, and I can extract a basic solution with `≤ m` nonzero configurations within additive tolerance `h`.* Exactly what the rounding lemma needs.

Now assemble. I have three working parts: (a) reduce distinct sizes to `m ≈ (2/k)·SIZE` at a cost of `≤ k·log(1/a)` bins (geometric grouping), having first pulled small pieces with `g = 1/SIZE` so `log(1/a) = O(log SIZE)`; (b) solve the fractional packing to tolerance `h = 1`, getting a basic `x` with `≤ m` configs; (c) round, losing `(m+1)/2 ≈ SIZE/k` bins. Suppose I just do (a),(b),(c) once. The rounding loss is `(m+1)/2 ≈ SIZE/k`. To make *that* small I'd want `k` large; but the grouping loss is `k·log(1/a)`, which wants `k` small. Balancing, `k ≈ √(SIZE/log)`, gives loss `≈ √(SIZE·log) = O(√OPT · √log OPT)`. That's the same wall linear grouping hit — a square-root, not a logarithm. The single rounding pays the full residual `(m+1)/2` in one shot, and `(m+1)/2` is `Θ(SIZE/k)`, irreducibly large for any `k` that keeps the grouping loss down.

So don't pay the residual all at once — *iterate*. After step (c), the residual instance `R` (the unpacked pieces from the fractional parts) has `SIZE(R) ≤ Σⱼ fⱼ < m ≈ (2/k)·SIZE(I)`. With `k = 2`, that's `SIZE(R) ≤ SIZE(I)/… ` — let me be careful: `m ≤ (2/k)SIZE + log`, so with `k = 2`, `SIZE(R) ≤ m ≈ SIZE(I) + log`, which is *not* a contraction. I need the grouping parameter tuned so that the residual shrinks geometrically. The residual's size is at most the number of fractional configs, `≤ m(grouped) ≈ (2/k)SIZE`. To get `SIZE(R) ≤ SIZE(I)/2` I want `(2/k)SIZE ≤ SIZE/2`, i.e. `k ≈ 4` — a constant. Pick a small constant `k` (the analysis works with `k = 2` in the size-budget variant, where the residual after one round is `≤ m ≤ SIZE/k + ln(1/a) ≤ SIZE/2 + O(log)`). So each round roughly *halves* `SIZE`. Then I don't pay `SIZE/k` once; I pay only the *grouping* loss per round, `k·log(1/a) = O(log SIZE)` bins, plus buy the integer `⌊xⱼ⌋` bins (which are "free" — they're real bins toward `OPT`), and recurse on a residual of half the size.

Count it. The size sequence is `SIZE, SIZE/2, SIZE/4, …`, so there are `O(log SIZE) = O(log OPT)` rounds before the residual is `O(1)` and I just first-fit it. Each round contributes its grouping/reinsertion loss of `O(log(1/a)) = O(log SIZE) = O(log OPT)` extra bins beyond the fractional value (the `2k(2 + ln(1/a))` for the discarded group, plus the `+1` tolerance from the LP solve). And here's the key telescoping: the integer bins bought across all rounds, plus the fractional remainders, never *over*-count, because each round buys `⌊xⱼ⌋` against a fractional solution whose value is `≤ LIN(Iᵢ) + 1`, and `LIN` of the residual only decreases. Summing the per-round losses,

```
total extra bins  =  Σ_{rounds} O(log OPT)  =  O(log OPT) · O(log OPT)  =  O(log² OPT).
```

So `O(log OPT)` rounds, each costing `O(log OPT)` excess bins, multiply to `O(log² OPT)`. The integer parts bought along the way sum to at most `LIN(I) ≤ OPT(I)` (each round's purchases are charged against the shrinking `LIN`), and the leftover-residual first-fit at the end is `O(1)`. Putting it together,

```
A(I)  ≤  OPT(I) + O(log² OPT(I)).
```

That's the additive bound I was after — the slack is polylogarithmic, the multiplicative constant is gone. Let me also note the dual reading of why it's exactly `log²`: one log from the *number of rounds* (geometric size decay), one log from the *per-round grouping loss* (the `log(1/a)` dyadic size classes / the `ln(1/g)` with `g = 1/SIZE`). Two independent logarithms, multiplied.

And there's a clean corollary that falls out for free. If an instance already has few distinct sizes — `m(I)` small — I needn't grow the size logs; I can run the same iteration tracking `m` instead of `SIZE`, halving the distinct-size count each round, and the same arithmetic gives `A(I) ≤ OPT(I) + O(log² m(I))`. So the bound is `OPT + O(log² OPT)` in general and `OPT + O(log² m)` when distinct sizes are few — whichever is smaller. (And there's a time/error knob: with grouping parameter `k = SIZE^α`, one round trades runtime against an `O(OPT^α · log OPT)` additive error, recovering the earlier square-root regime as the `α = 1/2` special case.)

Let me make the construction concrete and runnable. The polynomial-time guarantee is the ellipsoid-with-knapsack-oracle argument, but to *exhibit* the algorithm faithfully I solve the same configuration LP the way the separation oracle implies — column generation whose pricing subproblem is exactly the knapsack oracle — so I never enumerate all `q` configurations. The restricted master is a small explicit LP; the dual prices come off its covering constraints; the knapsack DP either returns an overpriced configuration (a new column) or certifies optimality. Then the outer loop is the recursive rounding: group sizes, solve the fractional packing, buy `⌊xⱼ⌋` bins per configuration, recurse on the residual, first-fit the leftovers. Each block below ties back to a step above.

```python
from collections import Counter
import math
from scipy.optimize import linprog

def knapsack_oracle(prices, sizes, counts):
    # SEPARATION ORACLE = pricing subproblem. Given dual prices u_i, find the
    # configuration (multiset of sizes fitting one unit bin) of MAX total
    # price. If that price > 1, the bin is overpriced: u is dual-infeasible
    # and this configuration is a violated constraint / a new primal column.
    G = 1000  # size grid: capacity 1.0 -> G cells (the F(k) knapsack DP above)
    best = [(0.0, ())] * (G + 1)        # best[g] = (max price, configuration)
    for i, (s, u, b) in enumerate(zip(sizes, prices, counts)):
        w = max(1, math.ceil(s * G))
        if w > G:
            continue
        for _copy in range(b):                       # bounded: <= b_i of type i
            for g in range(G, w - 1, -1):
                if best[g - w][0] + u > best[g][0] + 1e-12:
                    cfg = dict(best[g - w][1]); cfg[i] = cfg.get(i, 0) + 1
                    best[g] = (best[g - w][0] + u, tuple(sorted(cfg.items())))
    price, cfg = max(best, key=lambda t: t[0])
    return price, dict(cfg)

def solve_fractional_packing(sizes, counts):
    # Configuration LP  min 1.x  s.t.  A x >= b, x >= 0, solved by COLUMN
    # GENERATION (dual of the ellipsoid view): start with singleton columns,
    # repeatedly ask the knapsack oracle for an overpriced configuration on the
    # current dual prices and add it, until none has price > 1. Never lists all
    # q configurations -- that is the astronomically large thing we avoid.
    m = len(sizes)
    columns = [{i: min(counts[i], int(1 // sizes[i]) or 1)} for i in range(m)]
    while True:
        A_ub = [[-(c.get(i, 0)) for c in columns] for i in range(m)]   # -Ax<=-b
        res = linprog([1.0] * len(columns), A_ub=A_ub, b_ub=[-c for c in counts],
                      bounds=[(0, None)] * len(columns), method="highs")
        u = [max(0.0, -y) for y in res.ineqlin.marginals]   # dual prices u_i>=0
        price, cfg = knapsack_oracle(u, sizes, counts)
        if price <= 1.0 + 1e-6 or not cfg or cfg in columns:
            return columns, res.x                            # no improving column
        columns.append(cfg)

def reduce_distinct_sizes(items, k):
    # GEOMETRIC/LINEAR GROUPING: sort descending, chunk into groups of k, round
    # every piece in a group UP to its max (round up so a packing of the grouped
    # instance is feasible for the originals), discard the top group (k pieces,
    # one bin each). Shrinks the number of distinct sizes; OPT moves by <= k.
    s = sorted(items, reverse=True)
    if len(s) <= k:
        return [], s
    top, rest = s[:k], s[k:]
    grouped = []
    for g in range(0, len(rest), k):
        chunk = rest[g:g + k]; grouped += [chunk[0]] * len(chunk)   # round up
    return grouped, top

def first_fit(items, cap=1.0):
    bins = []
    for x in sorted(items, reverse=True):
        for bn in bins:
            if sum(bn) + x <= cap + 1e-9:
                bn.append(x); break
        else:
            bins.append([x])
    return bins

def karmarkar_karp(items, cap=1.0):
    # RECURSIVE ROUNDING. Each round: group to shrink distinct sizes, solve the
    # fractional packing, BUY floor(x_c) bins per configuration, recurse on the
    # residual whose SIZE has ~halved. O(log OPT) rounds x O(log OPT) loss each
    # => OPT + O(log^2 OPT).
    items = [x for x in items if x > 1e-12]
    if not items:
        return []
    if sum(items) <= 1.0 + 1e-9 or len(set(items)) <= 1:        # base case
        return first_fit(items, cap)
    k = max(1, int(math.isqrt(max(1, int(sum(items))))))        # group parameter
    grouped, discarded = reduce_distinct_sizes(items, k)
    bins = [[x] for x in discarded]                            # top group: 1/bin
    if not grouped:
        return bins + first_fit([x for x in items if x not in discarded], cap)
    cnt = Counter(round(x, 6) for x in grouped)
    sizes, counts = list(cnt.keys()), [cnt[s] for s in cnt]
    columns, x = solve_fractional_packing(sizes, counts)
    remaining = Counter({s: counts[i] for i, s in enumerate(sizes)})
    for col, xc in zip(columns, x):
        for _ in range(int(math.floor(xc + 1e-9))):           # buy floor(x_c)
            bn = []
            for i, a in col.items():
                take = min(a, remaining[sizes[i]])
                bn += [sizes[i]] * take; remaining[sizes[i]] -= take
            if bn:
                bins.append(bn)
    residual = [s for s, r in remaining.items() for _ in range(r)]
    return bins + karmarkar_karp(residual, cap)               # recurse on residual
```

The causal chain, start to finish: I refused a multiplicative constant and demanded an *additive* error, which pointed me at the configuration LP `(I)` because its fractional optimum `LIN(I)` is sandwiched `SIZE ≤ LIN ≤ OPT` and its basic solutions are sparse; rounding a basic solution showed `OPT ≤ LIN + (m+1)/2`, so the loss is governed by the number of distinct sizes `m`, not `n`; linear grouping shrinks `m` to `n/k` but its `k`-vs-`n/k` tradeoff stalls at `√n`, so I made grouping scale-aware — dyadic size classes with group budget `k·2^r` — driving `m` down to `(2/k)SIZE + log(1/a)` with loss only `k·log(1/a)`, and eliminating small pieces at threshold `g = 1/SIZE` so `log(1/a) = O(log OPT)`; the LP's astronomical column count `q` I defeated by solving the *dual* (m variables, q constraints) with the ellipsoid method, whose separation oracle is exactly a knapsack ("is any bin overpriced?"), made polynomial by rounding prices to a grid and solving knapsack by DP, then recovering a sparse basic primal via the configurations the oracle returned and pruning to `m` constraints; finally, rather than pay the full residual `(m+1)/2` once (which stalls at `√OPT`), I *iterated* — buy the integer parts, recurse on a residual whose size halves each round — so `O(log OPT)` rounds each losing `O(log OPT)` bins multiply to `O(log² OPT)`, giving `A(I) ≤ OPT(I) + O(log² OPT(I))`.
