## Research question

A mill keeps standard stock lengths of a material — rolls of paper of width `L`, bars, sheets — in unlimited supply, each carrying a cost. Customers send in an *order*: a list of `m` requested lengths `ℓ₁, …, ℓ_m`, with `Nᵢ` pieces wanted of length `ℓᵢ`. To fill the order the mill repeatedly takes a stock length and cuts it crosswise into ordered pieces; whatever is left over is trim loss. The problem is to fill the whole order at minimum total cost of consumed stock (equivalently, when all stock lengths and costs are equal, to use the fewest stock pieces, i.e. minimize trim waste).

The difficulty is combinatorial. A single way of cutting one stock length into ordered pieces — say, from a length-17 bar, one piece of 5 and two of 4 — is one *cutting pattern*. The number of distinct feasible patterns explodes with the number of requested lengths and the width of the stock: a pattern is any non-negative integer packing of the requested lengths into the stock width, and there are astronomically many of these. A solution method has to choose how many times to run each pattern without ever being able to list the patterns. A workable method would have to produce a low-cost cutting plan and certify that no un-listed pattern could do better.

## Background

**The trim problem and the activity (pattern) formulation.** Eisemann's "trim problem" (Management Science, 1957) had already framed paper-trim loss as a linear program whose *activities* are cutting patterns. The natural decision model is: index every feasible pattern `j`; let `xⱼ` be how many times pattern `j` is run; let `aᵢⱼ` be the number of pieces of length `ℓᵢ` that pattern `j` produces. Then a pattern is feasible iff it fits the stock width, `Σᵢ aᵢⱼ ℓᵢ ≤ L`, and the program is `min Σⱼ cⱼ xⱼ` subject to `Σⱼ aᵢⱼ xⱼ ≥ Nᵢ` for every requested length, `xⱼ ≥ 0` integer (`cⱼ` is the cost of the stock length pattern `j` cuts from). The model is exact and tiny to *state per column* — but the column index set is the set of all feasible patterns, which is enormous.

**Why the model is intractable, stated precisely.** Two factors make this formulation impractical. First, the number `n` of pattern variables is huge — it grows combinatorially with the number `k` of stock lengths and the number `m` of requested lengths, so for any realistic order one cannot even write down, let alone store, the constraint matrix. Second, the variables are restricted to integers, and integer programs are hard in their own right. These two are separable: the integrality is a familiar obstacle handled by rounding or branching, whereas the sheer column count is the obstacle peculiar to this problem.

**The LP relaxation and the simplex method.** Dropping the integrality restriction gives a linear program. For an order with large piece counts `Nᵢ`, an LP-optimal cutting plan is generally fractional, but rounding or solving a final integer master over the useful patterns changes only a small fraction of a large order in the intended regime. The LP cost is a valid lower bound, so it gives an a posteriori certificate for the integer plan's gap. This makes the LP relaxation the right thing to solve first. The simplex method walks from one basic feasible solution (a choice of `m` basic columns forming an invertible basis `B`) to a better neighbor. Its inner loop is *pricing*: it scans the non-basic columns for one whose entry would reduce the objective, brings it into the basis, and repeats until none remains. A standard simplification: when integrality is dropped, the demand slack variables can be removed, because any solution that over-fills a demand has an equal-cost solution that does not — so one may work with equality and `m` activities, or keep slacks (which can let a minimal solution use fewer than `m` activities, which helps the eventual rounding).

**Reduced cost and LP duality.** The pricing test is the reduced cost. Given a basis `B` with cost row `c_B`, the simplex multipliers (dual prices) are `π = c_B B⁻¹`, one price `πᵢ` per demand row. A non-basic column `P = (a₁, …, a_m)` with direct cost `c` has reduced cost `c − πᵀP`; the column can improve the solution exactly when this is negative, i.e. when `πᵀP > c`. The vector `B⁻¹P` (the representation of the new column in the current basis) is computed as part of every simplex iteration, so the multipliers `π` are always on hand. The optimality certificate is dual feasibility: if no column has negative reduced cost — `πᵀP ≤ c` for *every* feasible column — the current basic solution is LP-optimal. This certificate refers to *all* columns, including the ones never listed.

**The knapsack problem and its dynamic-programming solution.** The knapsack problem packs items of given "size" and "value" into a capacity to maximize total value. Dantzig had given both a fast greedy rule (take items in decreasing value-to-size ratio) and an exact dynamic program. In the integer unbounded form, after measuring lengths on an integer grid, define `F_s(x)` as the best achievable value with the first `s` item types, capacity `x`, sizes `ℓ₁, …, ℓ_s`, and values `b₁, …, b_s`. Then `F_{s+1}(x) = max_{0 ≤ r ≤ ⌊x/ℓ_{s+1}⌋} { r·b_{s+1} + F_s(x − r ℓ_{s+1}) }`, filling a table up to the capacity. One pass to the largest capacity yields the optima for all smaller capacities at once.

**Prior LP work on structured, large problems.** Ford and Fulkerson (Management Science, 1958), attacking maximal multi-commodity network flows, proposed a "specialized computing scheme that takes advantage of the structure" of a problem whose natural LP has one column per path. Dantzig and Wolfe (1960) gave a decomposition principle for linear programs with block structure: a master LP coupled to a subsystem that has its own LP. Both are part of the prior art on solving linear programs whose explicit form is unmanageably large.

## Baselines

**Direct integer/LP over all patterns (Eisemann-style trim LP).** Write the full pattern formulation and hand it to an LP or IP solver. Core idea: enumerate every feasible cutting pattern as a column, solve `min cᵀx` s.t. `Ax ≥ N`. Gap: the column set is combinatorially large, so for realistic orders the matrix cannot be formed or stored; the method does not scale past toy instances. This is the obstacle everything else is reacting to.

**Rounding / ad-hoc trim heuristics.** Greedily build patterns to cover the order — e.g. repeatedly cut the longest still-needed piece and fill the remainder of the stock length greedily. Core idea: avoid the LP entirely by constructing patterns directly. Gap: no optimality guarantee and no lower bound; trim loss can be far from the achievable minimum, and there is no certificate to say how far off.

**Kantorovich's roll-indexed compact formulation (1939; translated 1960).** Bound the number of stock pieces used by `κ`, give each potential roll `k` its own variables: a binary `x_{k0}` (roll used or not) and integer `x_{ki}` (how many of item `i` cut from roll `k`); `min Σ_k x_{k0}` s.t. `Σ_k x_{ki} ≥ Nᵢ` and `Σᵢ ℓᵢ x_{ki} ≤ L x_{k0}`. Core idea: a polynomial-size model with no pattern enumeration. Gap: its linear relaxation is weak — the LP value collapses to `Σᵢ ℓᵢ Nᵢ / L`, the pure material-area bound, ignoring that pieces must be cut in integer combinations — and the rolls are interchangeable, so the model is riddled with symmetry that makes branch-and-bound explore many redundant, identical subtrees.

## Evaluation settings

The natural instance is specified by a set of stock lengths `L₁, …, L_k` with costs `c₁, …, c_k`, and an order list of requested lengths `ℓ₁, …, ℓ_m` with demands `N₁, …, N_m`. A small illustrative instance has stock lengths `5, 6, 9` of cost `6, 7, 10` and an order of `20` pieces of length `2`, `10` of length `3`, `20` of length `4`. The quantities of interest are the LP-relaxation cost (a lower bound on integer cost), the number of distinct patterns the method must touch before certifying optimality (versus the total number of feasible patterns), and, after rounding/branching, the integer cost and the trim loss. Industrial paper-trim instances feature one or few stock widths and tens of ordered widths with large demand counts, the regime where the LP relaxation is nearly integral and the column count is hopeless to enumerate.

## Code framework

The existing primitives are an LP solver exposing both the primal optimum and the dual prices on each constraint, such as `scipy.optimize.linprog` with `ineqlin.marginals`, plus an integer optimizer for small packing subproblems.

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# Instance: stock width L, item lengths, demands.
# L: float ; lengths: array[m] ; demand: array[m]

def solve_cutting_stock(stock_len, lengths, demand):
    # Return a minimum-cost cutting plan for the order.
    # TODO
    pass
```
