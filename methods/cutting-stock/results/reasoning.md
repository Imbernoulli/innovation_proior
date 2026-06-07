Let me start from the thing that's actually blocking me, which is that I have a clean, exact model of the cutting-stock problem and I literally cannot write it down.

A mill stocks standard lengths — say rolls of width `L` — and an order comes in for `Nᵢ` pieces of each requested length `ℓᵢ`, `i = 1…m`. To fill the order I take stock pieces and slice each one crosswise into ordered lengths. One specific way of slicing one stock piece is a *pattern*: from a length-9 roll I could cut three pieces of length 3, or two of 4 plus one of nothing-useful, and so on. A pattern is just a list of how many of each ordered length it yields, `(a₁, …, a_m)`, and it's feasible exactly when the pieces fit, `Σᵢ aᵢ ℓᵢ ≤ L`. If I let `xⱼ` be the number of times I run pattern `j`, and `cⱼ` be the cost of the stock length it cuts from, the problem is

`min Σⱼ cⱼ xⱼ` subject to `Σⱼ aᵢⱼ xⱼ ≥ Nᵢ` for each `i`, with `xⱼ ≥ 0` integer.

That's it. It's correct, it's an integer program, and it's hopeless — not because it's wrong but because of `j`. The patterns are *every* feasible integer packing of the requested lengths into the width `L`. For a width of any size and a dozen requested lengths there are astronomically many of them. I can't enumerate the columns, can't form the matrix, can't store it. So the obstacle here is not the usual one. There are really two separable troubles: the integrality, which is the familiar hard-but-known obstacle, and the column count `n`, which is the obstacle peculiar to this problem. Let me deal with them one at a time, and let me get the easy one out of the way first because I suspect it's a distraction.

Take the integrality. Suppose I just drop it and solve the LP relaxation, `xⱼ ≥ 0` real. What do I lose? In a real order the demand counts `Nᵢ` are large — hundreds or thousands of pieces — so the LP-optimal `xⱼ` usually come out large too, and rounding or re-solving a final integer master over the useful patterns perturbs only a small fraction of a large order. The LP cost is a genuine lower bound on the integer cost, since I only enlarged the feasible set by relaxing, so it gives me a way to measure the gap of whatever integer plan I build afterward. If I need an exact integer proof, I can branch after the LP machinery is in place. Either way the integers are an afterward — a rounding, restricted integer master, or branching step bolted onto an LP solve. Good. So the real problem, the whole problem, is: solve that LP when I can't write its columns down.

There is even a small bonus from dropping integrality. With integers, if a solution over-fills some demand I'm stuck with slack variables in the formulation. But in the LP, any solution that over-fills a demand can be massaged — replace an offending pattern by the identical pattern that simply scraps the surplus piece, same cost — into a solution that meets demand exactly, with no slack and the same cost. So I can run with equality constraints and pure pattern columns. (I might *keep* slacks anyway, because then a minimal LP solution can use fewer than `m` patterns, which gives the eventual rounding a little more room. But conceptually they drop out.) Fine — the relaxation is clean. On to the wall.

So: an LP with an unwritable number of columns. My first instinct is to ask whether I even *need* all the columns. Let me look hard at what the simplex method actually does, because I have a feeling it touches far fewer columns than it carries.

Simplex moves between basic feasible solutions. A basis is a choice of `m` columns `P₁…P_m` forming an invertible matrix `B`; the basic solution reads off `B⁻¹` times the demand vector. To improve, simplex *prices*: it looks for a non-basic column whose entry into the basis would lower the cost. The test is the reduced cost. With cost row `c_B` on the basic columns, define the dual prices `π = c_B B⁻¹` — one price `πᵢ` per demand row. Then a non-basic column `P = (a₁,…,a_m)` with direct cost `c` has reduced cost `c − πᵀP`, and it's worth bringing in exactly when that's negative, i.e. when

`πᵀP > c`.

`B⁻¹` and hence `π` get recomputed every iteration anyway — the prices are always sitting there as a byproduct. The textbook simplex *uses* the prices by scanning a stored list of columns and computing `c − πᵀP` for each. But scanning is the only reason I thought I needed the list. The decision simplex makes at each step is tiny: it needs *one* column with negative reduced cost (to keep improving), or the fact that *no* column has one (to declare optimality). It never needs the columns it isn't moving. The list is just the haystack I rummage through to find the one needle.

So the question is: can I find that needle without owning the haystack? I have the prices `π`. A column is a feasible pattern `(a₁,…,a_m)` with `Σᵢ aᵢ ℓᵢ ≤ L`, and its direct cost `c` is the cost of the stock length. I want a feasible pattern with `πᵀP > c`. Instead of *checking* this against every pattern, what if I *search* for the pattern that maximizes `πᵀP`? If even the best pattern fails `πᵀ P > c`, then no pattern passes, and I'm optimal. If the best pattern passes, I've got my needle — and the most attractive one at that. So replace "scan all columns" with

maximize `πᵀP = Σᵢ πᵢ aᵢ` over all feasible patterns, i.e. `aᵢ ≥ 0` integer with `Σᵢ aᵢ ℓᵢ ≤ L`.

Let me stare at what I just wrote, because it looks awfully familiar. I'm choosing non-negative integer counts `aᵢ` of items, each item `i` has a "size" `ℓᵢ` and a "value" `πᵢ`, and I'm maximizing total value subject to total size not exceeding the capacity `L`. That's a knapsack problem. The pricing step — the thing I was dreading because it seemed to require all the columns — *is* a knapsack problem, one I can solve directly from the dual prices and the item lengths, with no column list at all.

That reframes the whole method. I don't enumerate columns; I generate a useful one on demand by solving a knapsack. Start with a few patterns, just enough to have a feasible basis. Solve the LP over only those — call it the restricted master. Read off the prices `π`. Solve the knapsack `max Σ πᵢ aᵢ s.t. Σ ℓᵢ aᵢ ≤ L`. If the knapsack value beats the stock cost, that pattern has negative reduced cost: add it as a new column and re-solve the restricted master. If it doesn't beat the cost — for any stock length — then no pattern anywhere has negative reduced cost, and the restricted LP solution is optimal *for the full, un-enumerated LP*. I get the optimum of a program I never wrote down. In a simplex-tableau implementation the active basis still has only `m` pattern columns; in a solver implementation I may keep a small generated pool, but that pool is never the full pattern universe.

I should sanity-check that this "generate instead of scan" move is legitimate and not a trick I'm fooling myself with. It's actually the same idea two adjacent problems already used. Ford and Fulkerson (1958), on multi-commodity flows, had a master LP whose columns were *paths*; rather than carry all path-columns they solved a shortest-path subproblem to produce the next useful one. Dantzig and Wolfe (1960) made the general statement: an LP with block structure can be run as a master over a few extreme points of a subsystem, with the subsystem's own optimization pricing in new extreme points. In both, the pricing scan over an implicit column set is replaced by an optimization that *constructs* the most attractive column. My situation is exactly this shape, and the subsystem I have to optimize happens to be a knapsack. So the move is sound; what's specific here is that the column-generating subproblem is of knapsack type, which is a piece of luck because knapsacks I can solve.

Let me now nail the reduced-cost condition exactly, in the original cost-`c` form, so I don't fumble a sign. Take a basis of patterns `P₁…P_m` with costs `c₁…c_m`; let `A = [P₁ … P_m]` (the `m×m` basis), and `C = (c₁,…,c_m)`. A candidate new activity `P = (a₁,…,a_m)` cuts from a stock length of cost `c`. Express it in the basis: `A·U = P`, so `U = A⁻¹P`. Bringing `P` in improves the cost iff its cost is less than the cost of the basic representation it replaces, i.e. iff `C·U > c`. Now `C·U = C·A⁻¹·P = (C A⁻¹)·P`, and `C A⁻¹` is precisely the row of prices — call it `(b₁,…,b_m)`. So a profitable activity cutting from `L` exists iff there are non-negative integers `aᵢ` with

`Σᵢ aᵢ ℓᵢ ≤ L` (it fits the stock) and `Σᵢ bᵢ aᵢ > c` (it pays).

And `C A⁻¹` is on hand as part of normal simplex. So to test all of them at once: maximize `Σ bᵢ aᵢ` subject to `Σ ℓᵢ aᵢ ≤ L`; if the max exceeds `c`, a profitable pattern exists, otherwise (sweeping every stock length and its cost) the current solution is a minimum. Same knapsack, signs straight.

When the stock lengths are interchangeable and I just want the *fewest rolls*, this simplifies and it's worth writing in that form because it's the one I'll code. Every pattern costs `1` (one roll), so `min Σⱼ xⱼ` s.t. `Σⱼ aᵢⱼ xⱼ ≥ Nᵢ`. Give each demand row a dual price `πᵢ`. By LP duality the dual is `max Σᵢ Nᵢ πᵢ` s.t. `Σᵢ aᵢⱼ πᵢ ≤ 1` for every pattern `j`, `πᵢ ≥ 0` (the `πᵢ` come out non-negative precisely because I wrote demand as `≥`, and one checks every optimal solution respects that — these are valid dual inequalities). The reduced cost of a pattern with counts `(a₁,…,a_m)` is `1 − Σᵢ πᵢ aᵢ`. So the pricing knapsack is

`max Σᵢ πᵢ aᵢ s.t. Σᵢ ℓᵢ aᵢ ≤ L, aᵢ ≥ 0 integer`,

and a pattern has negative reduced cost iff this maximum exceeds `1`. Stop when it's `≤ 1`. Same object, cleaner numbers.

I should pause on *why patterns are the right columns at all*, because a tempting alternative has to be ruled out rather than ignored. Kantorovich's older model indexes by roll: a binary "roll used" per roll and integer "how many of item `i` from roll `k`," `min Σ_k x_{k0}` with `Σ_k x_{ki} ≥ Nᵢ` and `Σᵢ ℓᵢ x_{ki} ≤ L x_{k0}`. It's polynomial-size — no pattern enumeration — so why am I going the hard way? Because its LP relaxation is worthless. Relax the binaries and the LP just spreads material smoothly; the optimal LP value collapses to `Σᵢ ℓᵢ Nᵢ / L`, the pure total-area-over-width bound, which knows nothing about the fact that pieces come in indivisible integer combinations. And every roll is interchangeable, so the model is drowning in symmetry — branch-and-bound would chew through endless identical subtrees. The pattern model is the opposite: each column already *is* a feasible integer packing, so the LP relaxation "knows" about integrality at the pattern level and gives a much tighter bound. The price of that tightness is the exponential column count — which is exactly the price column generation refuses to pay up front and pays only on demand. So the reformulation into patterns isn't gratuitous; it's bought a strong bound, and column generation is what makes the strong bound affordable.

Now I need to actually *solve* the pricing knapsack, repeatedly, fast. Two methods, and I'll use the cheap one first. The greedy rule (Dantzig's): sort items by value-to-size ratio `bᵢ/ℓᵢ` descending, then fill greedily — take `⌊L/ℓ_{i₁}⌋` of the best item, then `⌊(remaining)/ℓ_{i₂}⌋` of the next, and so on down the list. It's nearly free and very often hands me a profitable pattern outright. Only when greedy fails to produce a pattern that beats the cost — for *every* stock length — do I need to be sure none exists, and for that I want the exact knapsack. If the lengths are not already integral, I first measure them in the smallest common unit used by the order; the dynamic program is over that integer capacity grid.

Exact knapsack by dynamic programming. Let `F_s(x)` be the best value `Σᵢ₌₁ˢ bᵢ aᵢ` achievable using only the first `s` item types within capacity `x`. The recursion is: for the `(s+1)`-th item I decide how many copies `r` of it to take, `0 ≤ r ≤ ⌊x/ℓ_{s+1}⌋`, and use the best arrangement of earlier items in what's left:

`F_{s+1}(x) = max_{0 ≤ r ≤ ⌊x/ℓ_{s+1}⌋} { r·b_{s+1} + F_s(x − r ℓ_{s+1}) }`.

Fill the table up to the largest stock length `L_max`, and a single pass gives me `F_m(L)` for every smaller stock length for free along the way — one dynamic program prices in a new column for *all* stock lengths at once. So my pricing routine is: try greedy for each stock length; if none yields a beating pattern, run one DP up to the largest stock length and read off whether any stock length admits a profitable pattern, and if so which.

I still need a starting basis — a few feasible patterns to seed the restricted master before any prices exist. The cleanest seed: one "homogeneous" pattern per requested length. For item `i`, pick a stock length `Lⱼ ≥ ℓᵢ` and cut it into as many copies of `ℓᵢ` as fit, `⌊Lⱼ/ℓᵢ⌋` of them, scrapping the rest. That gives `m` patterns, each producing exactly one item type, so the pattern matrix is diagonal — obviously invertible, obviously feasible (run each enough times to meet its own demand). It's wasteful as a *solution*, but it's a legitimate basis to start simplex from, and the prices it produces will immediately pull in better, mixed patterns.

I want one concrete hand check before I code, because this is where sign mistakes hide. Take an order of 20 pieces of length 2, 10 of length 3, and 20 of length 4, with stock lengths 5, 6, and 9 costing 6, 7, and 10. Try the longest stock first because it gives the richest feasible patterns. The first pricing inequality for length 9 is `2a₁ + 3a₂ + 4a₃ ≤ 9`, and the current multiplier inequality asks whether the priced value can exceed 10. Greedy gives `(0,3,0)`, three 3's in a length-9 stock piece, so that column enters. After the tableau changes, a later length-9 price admits `(4,0,0)`, four 2's using 8 of the 9 units; then a length-9 pattern `(0,0,2)` enters; then stock length 6 yields mixed patterns such as `(1,0,1)`. Eventually the greedy test fails on every stock length, so I have to do the exact DP. With the current multipliers, the table values at capacities 5, 6, and 9 show no stock-5 or stock-6 improvement, but a stock-9 pattern `(1,1,1)` still beats cost 10, so that column enters too. After one more price update, the exact DP values at the three stock lengths are at or below their costs, so the reduced costs are all non-negative. The basis then gives cost 170: 10 stock pieces of length 6 cut as one 4 and one 2, and 10 stock pieces of length 9 cut as one 2, one 3, and one 4. The demands add up exactly. The integrality is a fortunate property of this arithmetic check, not something the LP relaxation promises.

One practical wrinkle follows from using a restricted master: the prices do not have to settle smoothly. Early on the restricted master has too few columns to give stable duals, so the first generated patterns can be poor, and the prices may swing before homing in. The bare loop still has the simplex optimality test, and dual stabilization can be added later if the swings become expensive.

The objects I have just derived force the implementation: a restricted master returns duals; an integer knapsack turns those duals into a pattern; the loop appends every pattern whose reduced cost is below zero; then a final integer master over the generated patterns builds the cutting plan. I'll use the one-stock-length, fewest-rolls form in code, since that is the common implementation case. If there are several stock lengths with different costs, the pricing call is repeated for each stock length and the best negative-reduced-cost column is appended with its own cost.

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

def solve_master(patterns, demand):
    """Restricted master LP: min rolls over the patterns generated so far."""
    A = np.asarray(patterns, dtype=float).T
    demand = np.asarray(demand, dtype=float)
    n_patterns = A.shape[1]
    res = linprog(
        np.ones(n_patterns),
        A_ub=-A,                    # -A x <= -demand  <=>  A x >= demand
        b_ub=-demand,
        bounds=[(0, None)] * n_patterns,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    duals = -res.ineqlin.marginals  # demand-row prices pi_i >= 0
    return float(res.fun), duals

def solve_pricing(stock_len, lengths, duals):
    """Integer knapsack pricing: find the pattern with smallest reduced cost."""
    lengths = np.asarray(lengths, dtype=float)
    ub = np.floor(stock_len / lengths)
    res = milp(
        c=-np.asarray(duals, dtype=float),              # maximize dual value
        constraints=LinearConstraint(lengths, -np.inf, stock_len),
        integrality=np.ones(len(lengths)),
        bounds=Bounds(0, ub),
    )
    if not res.success:
        raise RuntimeError(res.message)
    pattern = np.round(res.x).astype(int)
    reduced_cost = 1.0 - float(duals @ pattern)
    return pattern, reduced_cost

def column_generation(stock_len, lengths, demand, eps=1e-6):
    """Generate columns until no one-roll pattern has negative reduced cost."""
    lengths = np.asarray(lengths, dtype=float)
    patterns = []
    for i, length in enumerate(lengths):
        copies = int(stock_len // length)
        if copies <= 0:
            raise ValueError("each requested length must fit in the stock length")
        pattern = [0] * len(lengths)
        pattern[i] = copies
        patterns.append(pattern)

    while True:
        lp_value, duals = solve_master(patterns, demand)
        pattern, reduced_cost = solve_pricing(stock_len, lengths, duals)
        if reduced_cost >= -eps:
            return patterns, lp_value
        patterns.append(pattern.tolist())

def solve_integer(patterns, demand):
    """Final integer master over the generated columns."""
    A = np.asarray(patterns, dtype=float).T
    demand = np.asarray(demand, dtype=float)
    n_patterns = A.shape[1]
    res = milp(
        c=np.ones(n_patterns),
        constraints=LinearConstraint(A, demand, np.inf),
        integrality=np.ones(n_patterns),
        bounds=Bounds(0, np.inf),
    )
    if not res.success:
        raise RuntimeError(res.message)
    counts = np.round(res.x).astype(int)
    return counts, int(round(res.fun))

def solve_cutting_stock(stock_len, lengths, demand):
    patterns, lp_lower_bound = column_generation(stock_len, lengths, demand)
    counts, rolls = solve_integer(patterns, demand)
    plan = [(patterns[j], int(counts[j])) for j in range(len(patterns)) if counts[j] > 0]
    return {"lp_lower_bound": lp_lower_bound, "rolls": rolls, "plan": plan}
```

The causal chain, start to finish: the exact pattern model is correct but has astronomically many columns, so I can't even write the LP — but dropping integrality leaves a pure LP with a useful lower bound, and the simplex method that solves it only needs the single most-negative-reduced-cost column each step. Reduced cost `1 − Σ πᵢ aᵢ` is read straight off the duals, so the search for the best new column becomes "maximize priced value `Σ πᵢ aᵢ` under width `Σ ℓᵢ aᵢ ≤ L`" — a knapsack. So I never enumerate: solve the restricted master on a few patterns, read its duals, solve the knapsack to generate the best new pattern, add it, re-solve, and stop when even the best knapsack pattern cannot beat its cost. That certifies LP-optimality over all the patterns I never wrote down; an integer plan then comes from rounding, a restricted integer master over the generated patterns, or a full branch-and-price continuation if I need an exact integer certificate.
