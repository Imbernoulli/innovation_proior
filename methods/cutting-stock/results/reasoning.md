Let me start from the thing that's actually blocking me, which is that I have a clean, exact model of the cutting-stock problem and I literally cannot write it down.

Here's the model, and it's the honest one. A mill stocks standard lengths — say rolls of width `L` — and an order comes in for `Nᵢ` pieces of each requested length `ℓᵢ`, `i = 1…m`. To fill the order I take stock pieces and slice each one crosswise into ordered lengths. One specific way of slicing one stock piece is a *pattern*: from a length-9 roll I could cut three pieces of length 3, or two of 4 plus one of nothing-useful, and so on. A pattern is just a list of how many of each ordered length it yields, `(a₁, …, a_m)`, and it's feasible exactly when the pieces fit, `Σᵢ aᵢ ℓᵢ ≤ L`. If I let `xⱼ` be the number of times I run pattern `j`, and `cⱼ` be the cost of the stock length it cuts from, the problem is

`min Σⱼ cⱼ xⱼ` subject to `Σⱼ aᵢⱼ xⱼ ≥ Nᵢ` for each `i`, with `xⱼ ≥ 0` integer.

That's it. It's correct, it's an integer program, and it's hopeless — not because it's wrong but because of `j`. The patterns are *every* feasible integer packing of the requested lengths into the width `L`. For a width of any size and a dozen requested lengths there are astronomically many of them. I can't enumerate the columns, can't form the matrix, can't store it. So the obstacle here is not the usual one. There are really two separable troubles: the integrality, which is the familiar hard-but-known obstacle, and the column count `n`, which is the obstacle peculiar to this problem. Let me deal with them one at a time, and let me get the easy one out of the way first because I suspect it's a distraction.

Take the integrality. Suppose I just drop it and solve the LP relaxation, `xⱼ ≥ 0` real. What do I lose? In a real order the demand counts `Nᵢ` are large — hundreds or thousands of pieces — so the LP-optimal `xⱼ` come out large too, and large fractional values round to integers with only a tiny *fractional* change in cost. And the LP cost is a genuine lower bound on the integer cost, since I only enlarged the feasible set by relaxing. So if I round the LP answer up I get a feasible integer plan whose cost exceeds a valid lower bound by a small percentage. That's good enough, and even if I want to be exact later I can branch on the fractional variables. Either way the integers are an *afterward* — a rounding or branching step bolted onto an LP solve. Good. So the real problem, the whole problem, is: solve that LP when I can't write its columns down.

There's even a small bonus from dropping integrality. With integers, if a solution over-fills some demand I'm stuck with slack variables in the formulation. But in the LP, any solution that over-fills a demand can be massaged — replace an offending pattern by the identical pattern that simply scraps the surplus piece, same cost — into a solution that meets demand exactly, with no slack and the same cost. So I can run with equality constraints and pure pattern columns. (I might *keep* slacks anyway, because then a minimal LP solution can use fewer than `m` patterns, which gives the eventual rounding a little more room. But conceptually they drop out.) Fine — the relaxation is clean. On to the wall.

So: an LP with an unwritable number of columns. My first instinct is to ask whether I even *need* all the columns. Let me look hard at what the simplex method actually does, because I have a feeling it touches far fewer columns than it carries.

Simplex moves between basic feasible solutions. A basis is a choice of `m` columns `P₁…P_m` forming an invertible matrix `B`; the basic solution reads off `B⁻¹` times the demand vector. To improve, simplex *prices*: it looks for a non-basic column whose entry into the basis would lower the cost. The test is the reduced cost. With cost row `c_B` on the basic columns, define the dual prices `π = c_B B⁻¹` — one price `πᵢ` per demand row. Then a non-basic column `P = (a₁,…,a_m)` with direct cost `c` has reduced cost `c − πᵀP`, and it's worth bringing in exactly when that's negative, i.e. when

`πᵀP > c`.

And here's the thing I want to underline: `B⁻¹` and hence `π` get recomputed every iteration anyway — the prices are always sitting there as a byproduct. The textbook simplex *uses* the prices by scanning a stored list of columns and computing `c − πᵀP` for each. But scanning is the only reason I thought I needed the list. The decision simplex makes at each step is tiny: it needs *one* column with negative reduced cost (to keep improving), or the fact that *no* column has one (to declare optimality). It never needs the columns it isn't moving. The list is just the haystack I rummage through to find the one needle.

So the question is: can I find that needle without owning the haystack? I have the prices `π`. A column is a feasible pattern `(a₁,…,a_m)` with `Σᵢ aᵢ ℓᵢ ≤ L`, and its direct cost `c` is the cost of the stock length. I want a feasible pattern with `πᵀP > c`. Instead of *checking* this against every pattern, what if I *search* for the pattern that maximizes `πᵀP`? If even the best pattern fails `πᵀ P > c`, then no pattern passes, and I'm optimal. If the best pattern passes, I've got my needle — and the most attractive one at that. So replace "scan all columns" with

maximize `πᵀP = Σᵢ πᵢ aᵢ` over all feasible patterns, i.e. `aᵢ ≥ 0` integer with `Σᵢ aᵢ ℓᵢ ≤ L`.

Let me stare at what I just wrote, because it looks awfully familiar. I'm choosing non-negative integer counts `aᵢ` of items, each item `i` has a "size" `ℓᵢ` and a "value" `πᵢ`, and I'm maximizing total value subject to total size not exceeding the capacity `L`. That's a knapsack problem. The pricing step — the thing I was dreading because it seemed to require all the columns — *is* a knapsack problem, one I can solve directly from the dual prices and the item lengths, with no column list at all.

That reframes the whole method. I don't store columns; I *generate* the one column I need, on demand, by solving a knapsack. Start with a few patterns, just enough to have a feasible basis. Solve the LP over only those — call it the restricted master. Read off the prices `π`. Solve the knapsack `max Σ πᵢ aᵢ s.t. Σ ℓᵢ aᵢ ≤ L`. If the knapsack value beats the stock cost, that pattern has negative reduced cost: add it as a new column and re-solve the restricted master. If it doesn't beat the cost — for any stock length — then no pattern anywhere has negative reduced cost, and the restricted LP solution is optimal *for the full, un-enumerated LP*. I get the optimum of a program I never wrote down. The matrix I compute with never has more columns than rows.

I should sanity-check that this "generate instead of scan" move is legitimate and not a trick I'm fooling myself with. It's actually the same idea two adjacent problems already used. Ford and Fulkerson (1958), on multi-commodity flows, had a master LP whose columns were *paths*; rather than carry all path-columns they solved a shortest-path subproblem to produce the next useful one. Dantzig and Wolfe (1960) made the general statement: an LP with block structure can be run as a master over a few extreme points of a subsystem, with the subsystem's own optimization pricing in new extreme points. In both, the pricing scan over an implicit column set is replaced by an optimization that *constructs* the most attractive column. My situation is exactly this shape, and the subsystem I have to optimize happens to be a knapsack. So the move is sound; what's specific here is that the column-generating subproblem is of knapsack type, which is a piece of luck because knapsacks I can solve.

Let me now nail the reduced-cost condition exactly, in the original cost-`c` form, so I don't fumble a sign. Take a basis of patterns `P₁…P_m` with costs `c₁…c_m`; let `A = [P₁ … P_m]` (the `m×m` basis), and `C = (c₁,…,c_m)`. A candidate new activity `P = (a₁,…,a_m)` cuts from a stock length of cost `c`. Express it in the basis: `A·U = P`, so `U = A⁻¹P`. Bringing `P` in improves the cost iff its cost is less than the cost of the basic representation it replaces, i.e. iff `C·U > c`. Now `C·U = C·A⁻¹·P = (C A⁻¹)·P`, and `C A⁻¹` is precisely the row of prices — call it `(b₁,…,b_m)`. So a profitable activity cutting from `L` exists iff there are non-negative integers `aᵢ` with

`Σᵢ aᵢ ℓᵢ ≤ L` (it fits the stock) and `Σᵢ bᵢ aᵢ > c` (it pays).

And `C A⁻¹` is on hand as part of normal simplex. So to test all of them at once: maximize `Σ bᵢ aᵢ` subject to `Σ ℓᵢ aᵢ ≤ L`; if the max exceeds `c` there's a profitable pattern, otherwise (sweeping every stock length and its cost) the current solution is a minimum. Same knapsack, signs straight.

When the stock lengths are interchangeable and I just want the *fewest rolls*, this simplifies and it's worth writing in that form because it's the one I'll code. Every pattern costs `1` (one roll), so `min Σⱼ xⱼ` s.t. `Σⱼ aᵢⱼ xⱼ ≥ Nᵢ`. Give each demand row a dual price `πᵢ`. By LP duality the dual is `max Σᵢ Nᵢ πᵢ` s.t. `Σᵢ aᵢⱼ πᵢ ≤ 1` for every pattern `j`, `πᵢ ≥ 0` (the `πᵢ` come out non-negative precisely because I wrote demand as `≥`, and one checks every optimal solution respects that — these are valid dual inequalities). The reduced cost of a pattern with counts `(a₁,…,a_m)` is `1 − Σᵢ πᵢ aᵢ`. So the pricing knapsack is

`max Σᵢ πᵢ aᵢ s.t. Σᵢ ℓᵢ aᵢ ≤ L, aᵢ ≥ 0 integer`,

and a pattern has negative reduced cost iff this maximum exceeds `1`. Stop when it's `≤ 1`. Same object, cleaner numbers.

I should pause on *why patterns are the right columns at all*, because there's a tempting alternative that I want to rule out rather than ignore. Kantorovich's older model indexes by roll: a binary "roll used" per roll and integer "how many of item `i` from roll `k`," `min Σ_k x_{k0}` with `Σ_k x_{ki} ≥ Nᵢ` and `Σᵢ ℓᵢ x_{ki} ≤ L x_{k0}`. It's polynomial-size — no pattern enumeration — so why am I going the hard way? Because its LP relaxation is worthless. Relax the binaries and the LP just spreads material smoothly; the optimal LP value collapses to `Σᵢ ℓᵢ Nᵢ / L`, the pure total-area-over-width bound, which knows nothing about the fact that pieces come in indivisible integer combinations. And every roll is interchangeable, so the model is drowning in symmetry — branch-and-bound would chew through endless identical subtrees. The pattern model is the opposite: each column already *is* a feasible integer packing, so the LP relaxation "knows" about integrality at the pattern level and gives a much tighter bound. The price of that tightness is the exponential column count — which is exactly the price column generation refuses to pay up front and pays only on demand. So the reformulation into patterns isn't gratuitous; it's bought a strong bound, and column generation is what makes the strong bound affordable.

Now I need to actually *solve* the pricing knapsack, repeatedly, fast. Two methods, and I'll use the cheap one first. The greedy rule (Dantzig's): sort items by value-to-size ratio `bᵢ/ℓᵢ` descending, then fill greedily — take `⌊L/ℓ_{i₁}⌋` of the best item, then `⌊(remaining)/ℓ_{i₂}⌋` of the next, and so on down the list. It's nearly free and very often hands me a profitable pattern outright. Only when greedy fails to produce a pattern that beats the cost — for *every* stock length — do I need to be sure none exists, and for that I want the exact knapsack.

Exact knapsack by dynamic programming. Let `F_s(x)` be the best value `Σᵢ₌₁ˢ bᵢ aᵢ` achievable using only the first `s` item types within capacity `x`. The recursion is: for the `(s+1)`-th item I decide how many copies `r` of it to take, `0 ≤ r ≤ ⌊x/ℓ_{s+1}⌋`, and use the best arrangement of earlier items in what's left:

`F_{s+1}(x) = max_{0 ≤ r ≤ ⌊x/ℓ_{s+1}⌋} { r·b_{s+1} + F_s(x − r ℓ_{s+1}) }`.

Fill the table up to the largest stock length `L₁`, and a single pass gives me `F_m(L₂), …, F_m(L_k)` for free along the way — one dynamic program prices in a new column for *all* stock lengths at once. So my pricing routine is: try greedy for each stock length; if none yields a beating pattern, run one DP up to the largest stock length and read off whether any stock length admits a profitable pattern, and if so which.

I still need a starting basis — a few feasible patterns to seed the restricted master before any prices exist. The cleanest seed: one "homogeneous" pattern per requested length. For item `i`, pick a stock length `Lⱼ ≥ ℓᵢ` and cut it into as many copies of `ℓᵢ` as fit, `⌊Lⱼ/ℓᵢ⌋` of them, scrapping the rest. That gives `m` patterns, each producing exactly one item type, so the pattern matrix is diagonal — obviously invertible, obviously feasible (run each enough times to meet its own demand). It's wasteful as a *solution*, but it's a legitimate basis to start simplex from, and the prices it produces will immediately pull in better, mixed patterns.

Let me run the whole loop once, concretely, to make sure it terminates somewhere sensible. Order: 20 pieces of length 2, 10 of length 3, 20 of length 4. Stock lengths 5, 6, 9 with costs 6, 7, 10. I'll try stock lengths in decreasing size because the longest one admits the richest patterns. Seed with homogeneous patterns. Solve the restricted master, read prices. Now price: with the current prices I ask, for stock length 9, is there a pattern `2a₁ + 3a₂ + 4a₃ ≤ 9` whose priced value beats the cost? The greedy/ad-hoc search throws up patterns like `(0,3,0)` — three 3's in a 9 — and I bring it in; re-solve; new prices; price again; bring in `(4,0,0)` — four 2's in an 8 — and so on. Each profitable pattern I find shifts the prices, which changes which pattern is profitable next, and the master cost ratchets down. The loop keeps finding patterns from stock-9, then from stock-6, until the ad-hoc search fails on every stock length; at that point I fall to the DP to be certain, computing `F₃(5), F₃(6), F₃(9)`, and it confirms no pattern beats its cost anywhere. The restricted LP I'm holding is then optimal for the full problem. It lands at cost 170: cut 10 length-6 rolls each into one 4 and one 2, and 10 length-9 rolls each into one 2, one 3, and one 4. The pieces add up — 20 twos, 10 threes, 20 fours — and here the LP solution happened to come out in whole rolls, so no rounding was even needed. That the integers fell out is luck, not a guarantee; in general I'd round or branch the fractional plan. But the structure held: a handful of generated patterns, a knapsack at each step, and a certificate of optimality without ever listing the patterns.

Let me also be honest about one practical wrinkle I noticed while running it, because it affects the code. The prices don't settle smoothly — early on the restricted master has too few columns to give meaningful duals, so the first patterns generated can be junk, and the prices swing around before homing in near the end. It converges, but the early iterations are noisy. (One can damp this by boxing the duals near their current values, but that's a refinement; the bare loop converges.)

Now I'll write it. I'll keep the slots exactly as the structure dictates: build the restricted master LP and get its duals; solve the integer knapsack from those duals; loop, adding a column whenever the knapsack value beats 1; then a final integer solve. I'll use the fewest-rolls form (every pattern costs 1) since that's the common case.

```python
import numpy as np
from scipy.optimize import linprog

# Instance: a single stock width L, item lengths, integer demands.
def column_generation(L, lengths, demand, max_iter=1000, tol=1e-6):
    lengths = np.asarray(lengths, dtype=float)
    demand  = np.asarray(demand,  dtype=float)
    m = len(lengths)

    # --- seed: one homogeneous pattern per item (diagonal => feasible basis) ---
    # pattern i cuts floor(L / l_i) copies of item i and nothing else.
    A = np.diag(np.floor(L / lengths)).astype(float)   # columns are patterns
    cost = np.ones(A.shape[1])                          # one roll per pattern

    while True:
        # --- restricted master LP: min 1^T x  s.t.  A x >= demand,  x >= 0 ---
        # linprog minimizes c^T x with A_ub x <= b_ub, so negate to get >=.
        master = linprog(cost, A_ub=-A, b_ub=-demand, bounds=(0, None))
        duals = -master.ineqlin.marginals   # dual price pi_i on each demand row

        # --- pricing knapsack: max sum_i pi_i a_i  s.t. sum_i l_i a_i <= L ---
        # solved as a min of the negation, integer a_i >= 0.
        price = linprog(-duals,
                        A_ub=np.atleast_2d(lengths), b_ub=np.atleast_1d(L),
                        bounds=(0, None), integrality=1)
        new_pattern = np.round(price.x)
        reduced_cost = 1.0 + price.fun        # = 1 - sum_i pi_i a_i

        # negative reduced cost => the new pattern improves the LP; add it.
        if reduced_cost < -tol and len(cost) < master.x.size + max_iter:
            A = np.hstack((A, new_pattern.reshape(-1, 1)))
            cost = np.append(cost, 1.0)
        else:
            break   # no pattern beats its cost => restricted LP is the full LP optimum

    # --- round the fractional LP plan up to a feasible integer cutting plan ---
    x_int = np.ceil(master.x)
    return A, master.x, x_int, master.fun
```

For the exact integer knapsack and the duals I'm leaning on a solver — `scipy.optimize.linprog` with `integrality=1` does the small knapsack, and `ineqlin.marginals` hands me the demand-row duals. If I'd rather expose the duals and the integer subproblem directly, the same loop in OR-Tools reads the same: a continuous (CLP) master whose constraints give `dual_value()`, and an integer (SCIP/CBC) subproblem maximizing `Σ πᵢ aᵢ` under `Σ ℓᵢ aᵢ ≤ L`.

```python
from ortools.linear_solver import pywraplp

def solve_master(patterns, lengths, demand):
    s = pywraplp.Solver.CreateSolver('CLP')           # LP relaxation, gives duals
    x = [s.NumVar(0, s.infinity(), f'x{j}') for j in range(len(patterns))]
    rows = []
    for i in range(len(lengths)):
        c = s.RowConstraint(demand[i], s.infinity(), f'd{i}')   # sum_j a_ij x_j >= N_i
        for j, p in enumerate(patterns):
            c.SetCoefficient(x[j], p[i])
        rows.append(c)
    s.Minimize(s.Sum(x))                              # min number of rolls
    s.Solve()
    duals = [rows[i].dual_value() for i in range(len(lengths))]
    return [v.solution_value() for v in x], duals, s.Objective().Value()

def solve_pricing(L, lengths, duals):
    s = pywraplp.Solver.CreateSolver('SCIP')          # integer knapsack
    a = [s.IntVar(0, int(L // lengths[i]), f'a{i}') for i in range(len(lengths))]
    s.Add(s.Sum(lengths[i] * a[i] for i in range(len(lengths))) <= L)
    s.Maximize(s.Sum(duals[i] * a[i] for i in range(len(lengths))))   # max sum pi_i a_i
    s.Solve()
    pattern = [int(round(a[i].solution_value())) for i in range(len(lengths))]
    reduced_cost = 1.0 - s.Objective().Value()        # 1 - sum pi_i a_i
    return pattern, reduced_cost

def cut_stock(L, lengths, demand):
    patterns = [[ (int(L // lengths[i]) if k == i else 0) for i in range(len(lengths))]
                for k in range(len(lengths))]         # homogeneous seed
    while True:
        x, duals, obj = solve_master(patterns, lengths, demand)
        pattern, reduced_cost = solve_pricing(L, lengths, duals)
        if reduced_cost < -1e-6:                      # improving column found
            patterns.append(pattern)
        else:
            break                                     # LP optimal over all patterns
    return patterns, x, obj
```

The causal chain, start to finish: the exact pattern model is correct but has astronomically many columns, so I can't even write the LP — but dropping integrality (rounding later, cheaply, because demands are large) leaves a pure LP, and the simplex method that solves it never needs the column list, only the single most-negative-reduced-cost column each step. Reduced cost `1 − Σ πᵢ aᵢ` is read straight off the duals, so the search for the best new column becomes "maximize priced value `Σ πᵢ aᵢ` under width `Σ ℓᵢ aᵢ ≤ L`" — a knapsack. So I never enumerate: solve the restricted master on a few patterns, read its duals, solve the knapsack to *generate* the best new pattern, add it, re-solve, and stop when even the best knapsack pattern can't beat its cost — which certifies LP-optimality over all the patterns I never wrote down. Round or branch for integers, and the answer comes out of a matrix that never has more columns than it has rows.
