# Gilmore–Gomory column generation for the cutting-stock problem

## Problem

Stock material comes in standard lengths/widths `L₁,…,L_k` of cost `c₁,…,c_k` in unlimited supply. An order requests `Nᵢ` pieces of length `ℓᵢ`, `i = 1…m`. Each stock piece is cut crosswise into ordered lengths according to some *cutting pattern*; leftover is trim loss. Minimize the total cost of stock consumed (when stock is uniform, minimize the number of pieces used, i.e. trim waste).

A pattern is a vector of integer piece counts `(a₁,…,a_m)` feasible iff `Σᵢ aᵢ ℓᵢ ≤ L`. The exact model assigns one variable `xⱼ` per pattern,

`min Σⱼ cⱼ xⱼ  s.t.  Σⱼ aᵢⱼ xⱼ ≥ Nᵢ (∀i),  xⱼ ≥ 0 integer.`

The number of patterns is combinatorially large, so the constraint matrix cannot even be written down.

## Key idea — delayed (Dantzig–Wolfe) column generation

Solve the LP relaxation; round/branch for integers afterward (large demands ⇒ rounding cost is a small fraction of a valid lower bound). The simplex method solving that LP never needs the full column list — only, at each step, the column of most negative reduced cost, or a guarantee that none is negative. So **generate** that column on demand instead of storing all columns:

- **Restricted master (RMP).** Solve the LP over a small current set of patterns; obtain the dual price `πᵢ ≥ 0` on each demand row. In the fewest-rolls form (`cⱼ = 1`): `min Σⱼ xⱼ` s.t. `Σⱼ aᵢⱼ xⱼ ≥ Nᵢ`, dual `max Σᵢ Nᵢ πᵢ` s.t. `Σᵢ aᵢⱼ πᵢ ≤ 1` (∀ patterns), `πᵢ ≥ 0`.
- **Reduced cost.** Pattern `(a₁,…,a_m)` from stock cost `c` has reduced cost `c − Σᵢ πᵢ aᵢ` (in fewest-rolls form, `1 − Σᵢ πᵢ aᵢ`). It improves the LP iff this is negative, i.e. iff `Σᵢ πᵢ aᵢ > c`.
- **Pricing = knapsack.** The most attractive new pattern maximizes the priced value subject to the width:

  `max Σᵢ πᵢ aᵢ  s.t.  Σᵢ ℓᵢ aᵢ ≤ L,  aᵢ ≥ 0 integer.`

  If the optimum exceeds `c` (resp. `1`) for some stock length, that pattern has negative reduced cost — add its column and re-solve the RMP. If no stock length yields a value above its cost, the current RMP solution is optimal for the full, un-enumerated LP.
- **Solving the knapsack.** A greedy fill by decreasing value/size ratio `πᵢ/ℓᵢ` usually suffices; the exact fallback is dynamic programming, `F_{s+1}(x) = max_{0≤r≤⌊x/ℓ_{s+1}⌋} { r π_{s+1} + F_s(x − r ℓ_{s+1}) }`, where one pass to the largest stock length prices in all stock lengths at once.
- **Integers.** Round up the fractional LP plan, or branch, to recover an integer cutting plan.

The matrix worked with never has more columns than rows.

## Algorithm

1. Seed with `m` homogeneous patterns — pattern `i` cuts `⌊L/ℓᵢ⌋` copies of item `i` (diagonal ⇒ feasible, invertible basis).
2. Solve the RMP LP; read duals `π`.
3. Solve the pricing knapsack `max Σ πᵢ aᵢ` s.t. `Σ ℓᵢ aᵢ ≤ L`.
4. If reduced cost `< 0` (knapsack value `> 1`), append the new pattern and go to 2; else stop — LP optimal.
5. Round/branch to integers.

## Code

`scipy.optimize.linprog` version — LP master with dual marginals, integer knapsack pricing, the generation loop:

```python
import numpy as np
from scipy.optimize import linprog

def column_generation(L, lengths, demand, tol=1e-6):
    lengths = np.asarray(lengths, dtype=float)
    demand  = np.asarray(demand,  dtype=float)

    # seed: one homogeneous pattern per item -> diagonal, feasible basis
    A = np.diag(np.floor(L / lengths)).astype(float)   # columns = patterns
    cost = np.ones(A.shape[1])                          # one roll per pattern

    while True:
        # restricted master: min 1^T x  s.t.  A x >= demand,  x >= 0
        master = linprog(cost, A_ub=-A, b_ub=-demand, bounds=(0, None))
        duals = -master.ineqlin.marginals              # dual price pi_i per demand row

        # pricing knapsack: max sum_i pi_i a_i  s.t.  sum_i l_i a_i <= L, a_i in Z+
        price = linprog(-duals,
                        A_ub=np.atleast_2d(lengths), b_ub=np.atleast_1d(L),
                        bounds=(0, None), integrality=1)
        new_pattern  = np.round(price.x)
        reduced_cost = 1.0 + price.fun                 # = 1 - sum_i pi_i a_i

        if reduced_cost < -tol:                        # improving column
            A = np.hstack((A, new_pattern.reshape(-1, 1)))
            cost = np.append(cost, 1.0)
        else:
            break                                      # full-LP optimum reached

    x_int = np.ceil(master.x)                          # round up to integer plan
    return A, master.x, x_int, master.fun
```

OR-Tools `pywraplp` version — a continuous CLP master exposing `dual_value()`, an integer SCIP knapsack subproblem:

```python
from ortools.linear_solver import pywraplp

def solve_master(patterns, lengths, demand):
    s = pywraplp.Solver.CreateSolver('CLP')                       # LP relaxation -> duals
    x = [s.NumVar(0, s.infinity(), f'x{j}') for j in range(len(patterns))]
    rows = []
    for i in range(len(lengths)):
        c = s.RowConstraint(demand[i], s.infinity(), f'd{i}')     # sum_j a_ij x_j >= N_i
        for j, p in enumerate(patterns):
            c.SetCoefficient(x[j], p[i])
        rows.append(c)
    s.Minimize(s.Sum(x))                                          # min number of rolls
    s.Solve()
    duals = [rows[i].dual_value() for i in range(len(lengths))]
    return [v.solution_value() for v in x], duals, s.Objective().Value()

def solve_pricing(L, lengths, duals):
    s = pywraplp.Solver.CreateSolver('SCIP')                      # integer knapsack
    a = [s.IntVar(0, int(L // lengths[i]), f'a{i}') for i in range(len(lengths))]
    s.Add(s.Sum(lengths[i] * a[i] for i in range(len(lengths))) <= L)
    s.Maximize(s.Sum(duals[i] * a[i] for i in range(len(lengths))))  # max sum pi_i a_i
    s.Solve()
    pattern = [int(round(a[i].solution_value())) for i in range(len(lengths))]
    return pattern, 1.0 - s.Objective().Value()                   # reduced cost

def cut_stock(L, lengths, demand):
    patterns = [[(int(L // lengths[i]) if k == i else 0) for i in range(len(lengths))]
                for k in range(len(lengths))]                     # homogeneous seed
    while True:
        x, duals, obj = solve_master(patterns, lengths, demand)
        pattern, reduced_cost = solve_pricing(L, lengths, duals)
        if reduced_cost < -1e-6:
            patterns.append(pattern)
        else:
            break                                                 # LP optimal over all patterns
    return patterns, x, obj
```

## Why it works

The pattern formulation gives a tight LP relaxation (each column is itself a feasible integer packing), unlike the roll-indexed compact model whose LP bound collapses to the area ratio `Σ ℓᵢ Nᵢ / L` and is plagued by symmetry. The cost of that tightness is exponentially many columns — which delayed column generation never pays up front: it prices implicitly by solving a knapsack, touching only the few patterns the simplex method actually needs, and certifies optimality over all patterns when the knapsack can no longer beat the stock cost.
