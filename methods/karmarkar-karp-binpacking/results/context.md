# Context — Packing Items into the Fewest Unit Bins, and How Close to Optimal a Polynomial Algorithm Can Get

## Research question

An instance `I` is a list of `n` pieces, each with a size that is a rational number in `(0, 1]`. The task is to partition the pieces among bins of unit capacity, using as few bins as possible, subject to the constraint that the sizes of the pieces placed in any one bin sum to at most `1`. Write `OPT(I)` for the minimum number of bins. Deciding `OPT(I)` exactly is NP-hard, so the realistic goal is a polynomial-time algorithm `A` whose output `A(I)` is close to `OPT(I)`.

For more than a decade the only guarantees anyone could prove were *multiplicative*: bounds of the shape `A(I) ≤ C·OPT(I) + o(OPT(I))`, and a body of work between 1973 and 1980 pushed the best constant `C` down from `1.7` to roughly `1.18333`. A multiplicative guarantee, however small the constant, wastes a fixed *fraction* of the optimal bins forever; on an instance needing a million bins a `C = 1.18` algorithm can squander a hundred-eighty-thousand. The sharper question is whether the *additive* error can be made to grow slowly: is there a polynomial-time algorithm with

```
A(I) ≤ OPT(I) + o(OPT(I)),
```

and how slowly can the additive slack grow — `O(OPT^{1-δ})`, `O(log OPT)`, polylogarithmic? An algorithm achieving such a bound must, on every large instance, use essentially the optimal number of bins, with the excess only a lower-order term, while still being correct for all inputs and running in time polynomial in `n`.

## Background

**The cost of the relaxation, and what makes the problem hard.** Two crude facts bracket `OPT(I)`. Let `SIZE(I) = Σ sᵢ` be the total size of all pieces. No packing can use fewer than `SIZE(I)` bins (each bin holds at most one unit of size), so `OPT(I) ≥ SIZE(I)`. Conversely, any packing in which at most one bin is less than half full uses at most `2·SIZE(I) + 1` bins — pair up the half-empty bins. So `SIZE(I) ≤ OPT(I) ≤ 2·SIZE(I) + 1`. The whole difficulty lives in the factor-of-two gap between these bounds: closing it for arbitrary instances is exactly the NP-hard part.

**The line of approximation algorithms (the prevailing practice).** The dominant approach was simple greedy rules analyzed for worst-case ratio. First-Fit-Decreasing (FFD) sorts pieces in decreasing size and drops each into the first bin that can hold it; its asymptotic worst-case ratio is `11/9 ≈ 1.222`, i.e. a relative error of about `2/9`. Variants and more careful analyses lowered the constant toward `1.18`, but all shared the multiplicative form and a constant that would not go to `1`. These are `O(n log n)` and excellent in practice, but their guarantee is a fixed-fraction overhead.

**The departure: an asymptotic approximation scheme (Fernandez de la Vega and Lueker, 1981).** This work broke the multiplicative-constant ceiling. It produces, for any `ε > 0`, a packing of cost `≤ (1+ε)·OPT(I) + O(ε⁻²)` in time linear in `n` (with a constant blowing up in `1/ε`). It rests on three reusable techniques. (i) *Elimination of small pieces*: set aside every piece smaller than a threshold `g`, pack the rest, then reinsert the small pieces greedily, opening a new bin only when forced; the reinsertion inflates the count by at most a `(1+g)` factor plus one. (ii) *Linear grouping*: to cut the number of distinct piece sizes, sort the pieces in decreasing order, cut them into consecutive groups of `k`, and round every piece in a group up to the largest size in that group; discard the top group (its `k` pieces are packed one per bin). The rounded instance `J` has at most `n/k` distinct sizes, and `OPT(J) ≤ OPT(I) ≤ OPT(J) + k` — distinct sizes drop while the optimum barely moves. (iii) *Rounding a fractional packing*: once the number of distinct sizes is a constant, an instance with few distinct sizes can be solved by enumerating bin types and rounding. Letting `ε` depend on the instance (Johnson's observation) converts the scheme into an algorithm with `A(I) ≤ OPT(I) + o(OPT(I))` — but the additive slack obtained this way is a genuine power of `OPT`, e.g. `O(OPT^{1-δ})`, not polylogarithmic, and the runtime degrades badly as `ε → 0`.

**The set-cover / cutting-stock relaxation (Eisemann 1957; Gilmore and Gomory 1961).** A far stronger relaxation than greedy analysis treats packing as choosing how many bins of each *configuration* to use. Group the pieces into `m` distinct sizes `s₁,…,s_m` with `bᵢ` pieces of size `i`. A *configuration* is a multiset of sizes that fits in one bin (`Σ aᵢⱼ sᵢ ≤ 1`); let `A = (aᵢⱼ)` be the `m × q` matrix whose columns are all configurations, `q` astronomically large. The integer program "use `xⱼ ≥ 0` bins of configuration `j`, cover all pieces" is exactly bin packing; its LP relaxation,

```
minimize 1·x   subject to   A x ≥ b,   x ≥ 0,
```

was introduced by Eisemann for the trim problem and used by Gilmore and Gomory for industrial cutting-stock. Write `LIN(I)` for its optimum. Because an integer packing is a feasible integer `x`, `LIN(I) ≤ OPT(I)`; and because the constraints force `Σ sᵢ` to be covered, `SIZE(I) ≤ LIN(I)`. So `SIZE(I) ≤ LIN(I) ≤ OPT(I)`. The relaxation is strong: its gap to the integer optimum is what a good algorithm can hope to convert into a small additive error. Gilmore and Gomory showed the LP is solvable *in practice* despite its `q` columns by **column generation** (a.k.a. delayed column generation): keep a small "restricted master" with a few columns, solve it, read off dual prices, and find a new improving column by solving a *pricing subproblem* — which for this LP is a **knapsack problem** (the most-improving configuration is the highest-priced multiset of sizes fitting in a bin). They did not, however, prove the LP solvable in *polynomial* time.

**The number of distinct sizes controls the rounding loss.** A basic feasible solution to any LP has at most as many nonzero variables as it has constraints. The configuration LP has `m` constraints (one per distinct size), so a basic feasible `x` uses at most `m` distinct configurations with `xⱼ > 0`. Rounding such an `x` to an integer packing — take `⌊xⱼ⌋` bins of each configuration, then clean up the leftover fractional parts — loses an amount governed by `m`, not by `n`: the residual left by the fractional parts has total size at most `m`, and can be packed in at most about `(m+1)/2` extra bins. So `OPT(I) ≤ LIN(I) + (m+1)/2`. This is the pivot the whole subject turns on: *if the number of distinct sizes can be made small, the loss in passing from the fractional optimum to an integer packing is small.* The tension is that real instances have many distinct sizes.

**The ellipsoid method and separation oracles (Grötschel, Lovász, and Schrijver, 1981).** Khachiyan's ellipsoid method solves an LP in polynomial time; GLS sharpened it into a tool that handles an LP with *exponentially many constraints*, provided one has a polynomial-time **separation oracle**: given a candidate point, either certify it feasible or return a single violated constraint. The ellipsoid method maintains a shrinking ellipsoid guaranteed to contain the optimum; at each step it queries the oracle at the center and slices the ellipsoid with the returned hyperplane (an "optimality cut" if the center is feasible, a "feasibility cut" if it returns a violated constraint), shrinking the volume by a fixed factor each iteration. After polynomially many iterations the optimum is pinned within any tolerance. Crucially, the algorithm never needs the constraints written out — only the oracle. This is the machine that could in principle solve a linear program with an astronomical number of columns, if its dual's constraints (one per column) admit a fast separation oracle.

## Baselines

**First-Fit-Decreasing (FFD) and the greedy family.** Core idea: sort pieces descending, place each in the first bin that fits. Algorithm: `O(n log n)`. Guarantee: asymptotic ratio `11/9`, relative error `≈ 2/9`. Gap: a fixed multiplicative overhead that does not tend to `1`; no additive guarantee, no use of the LP structure. The yardstick any additive scheme must beat in the asymptotic sense.

**The asymptotic scheme of Fernandez de la Vega and Lueker (1981).** Core idea: eliminate small pieces, use linear grouping to reduce to a constant number `O(1/ε²)` of distinct sizes, solve the few-distinct-sizes instance, reinsert. Algorithm: for fixed `ε`, `O(n)` time (with a constant exponential in `1/ε`); cost `≤ (1+ε)·OPT(I) + O(ε⁻²)`. Gap: still multiplicative in form; pushing `ε → 0` to chase an additive bound makes the runtime explode and yields only `O(OPT^{1-δ})` additive slack, not polylogarithmic. Its three techniques (small-piece elimination, linear grouping, fractional rounding) are the right primitives, but linear grouping alone cannot make the number of distinct sizes small *and* keep the grouping loss small at the same time.

**The Gilmore–Gomory column-generation solver for the configuration LP (1961).** Core idea: solve the configuration LP by delayed column generation with a knapsack pricing subproblem. Algorithm: restricted master LP + repeated knapsack pricing until no improving column exists. Gap: shown to work well *in practice* but never proven polynomial-time — the number of columns generated, and the cost of solving knapsacks, were not bounded. The LP it solves is exactly the strong relaxation needed, but its polynomial-time status was open.

**Direct rounding of the configuration LP, applied once.** Core idea: solve the LP, take a basic feasible `x`, buy `⌊xⱼ⌋` bins per configuration, clean up the residual. Guarantee: `OPT(I) ≤ LIN(I) + (m+1)/2`. Gap: the loss `(m+1)/2` is linear in the number of distinct sizes; on instances with many sizes it is `Θ(OPT)`, no better than greedy. A single rounding step is too blunt — the loss must be made lower-order, either by shrinking `m` drastically or by not paying the full residual cost at once.

## Evaluation settings

The yardstick is the additive error `A(I) − OPT(I)` as a function of `OPT(I)` (equivalently of `SIZE(I)`, since `SIZE ≤ OPT ≤ 2·SIZE + 1`), measured against the lower bounds `SIZE(I) ≤ LIN(I) ≤ OPT(I)` and the greedy `11/9` baseline. Instances are `n` pieces of rational sizes in `(0,1]`, with `n(I)` the piece count, `m(I)` the number of distinct sizes, `a(I)` the smallest piece size, and `SIZE(I)` the total size. The relevant cost measures are the number of bins used and the running time, the latter expressed through a function `T(m, n)` bounding the time to solve the configuration LP to a given additive tolerance. The analytic instruments available are: LP duality (a configuration LP and its dual price LP have equal optimum), the basic-feasible-solution sparsity fact (`≤ m` nonzero variables), the separation-oracle / ellipsoid runtime accounting, dynamic-programming bounds for knapsack, and recursion/geometric-series sums for any scheme that reduces the instance and recurses. Correctness is exact: every piece is placed, no bin exceeds capacity. The regime of interest is large `OPT`, where the additive term dominates.

## Code framework

Available primitives: an LP solver for a small explicit linear program (the restricted master), a dynamic-programming knapsack routine (the pricing / separation subproblem), a first-fit packer for residual and small pieces, integer arithmetic for rounding fractional solutions, and counting/sorting utilities to manage distinct sizes. The scaffold is an outer routine that reduces an instance, calls a fractional-packing subroutine, rounds, and handles leftovers — with empty slots for the grouping rule, the way the huge LP is solved, and how the fractional solution is turned into an integer packing.

```python
from collections import Counter
import math

def knapsack_oracle(prices, sizes, counts):
    # pricing / separation subproblem: highest-priced multiset of sizes
    # fitting in a unit bin; if its price > 1 it is an improving column /
    # violated dual constraint.
    pass  # TODO

def solve_fractional_packing(sizes, counts):
    # solve  min 1.x  s.t.  A x >= b, x >= 0  over the (astronomically many)
    # bin configurations, returning a sparse near-optimal x.
    # TODO: how do we avoid materializing all configurations?
    pass

def reduce_distinct_sizes(items, k):
    # TODO: shrink the number of distinct sizes without moving OPT much.
    pass

def first_fit(items, cap=1.0):
    bins = []
    for x in sorted(items, reverse=True):
        for bn in bins:
            if sum(bn) + x <= cap + 1e-9:
                bn.append(x); break
        else:
            bins.append([x])
    return bins

def pack(items, cap=1.0):
    items = [x for x in items if x > 1e-12]
    if not items or sum(items) <= 1.0 + 1e-9:
        return first_fit(items, cap)
    # TODO: reduce distinct sizes; solve the fractional packing; turn the
    # fractional solution into an integer packing; recurse on what's left.
    return ...  # TODO
```
