# Gilmore–Gomory column generation for the cutting-stock problem

## Problem

Stock material comes in standard lengths/widths `L₁,…,L_k` of cost `c₁,…,c_k` in unlimited supply. An order requests `Nᵢ` pieces of length `ℓᵢ`, `i = 1…m`. Each stock piece is cut crosswise into ordered lengths according to some *cutting pattern*; leftover is trim loss. Minimize the total cost of stock consumed (when stock is uniform, minimize the number of pieces used, i.e. trim waste).

A pattern is a vector of integer piece counts `(a₁,…,a_m)` feasible iff `Σᵢ aᵢ ℓᵢ ≤ L`. The exact model assigns one variable `xⱼ` per pattern,

`min Σⱼ cⱼ xⱼ  s.t.  Σⱼ aᵢⱼ xⱼ ≥ Nᵢ (∀i),  xⱼ ≥ 0 integer.`

The number of patterns is combinatorially large, so the constraint matrix cannot even be written down.

## Key idea — delayed (Dantzig–Wolfe) column generation

Solve the LP relaxation first; it gives a lower bound, and integer recovery can then be done by rounding, by a final integer master over generated patterns, or by branching if an exact integer certificate is needed. The simplex method solving the LP never needs the full column list — only, at each step, the column of most negative reduced cost, or a guarantee that none is negative. So **generate** that column on demand instead of storing all columns:

- **Restricted master (RMP).** Solve the LP over a small current set of patterns; obtain the dual price `πᵢ ≥ 0` on each demand row. In the fewest-rolls form (`cⱼ = 1`): `min Σⱼ xⱼ` s.t. `Σⱼ aᵢⱼ xⱼ ≥ Nᵢ`, dual `max Σᵢ Nᵢ πᵢ` s.t. `Σᵢ aᵢⱼ πᵢ ≤ 1` (∀ patterns), `πᵢ ≥ 0`.
- **Reduced cost.** Pattern `(a₁,…,a_m)` from stock cost `c` has reduced cost `c − Σᵢ πᵢ aᵢ` (in fewest-rolls form, `1 − Σᵢ πᵢ aᵢ`). It improves the LP iff this is negative, i.e. iff `Σᵢ πᵢ aᵢ > c`.
- **Pricing = knapsack.** The most attractive new pattern maximizes the priced value subject to the width:

  `max Σᵢ πᵢ aᵢ  s.t.  Σᵢ ℓᵢ aᵢ ≤ L,  aᵢ ≥ 0 integer.`

  If the optimum exceeds `c` (resp. `1`) for some stock length, that pattern has negative reduced cost — add its column and re-solve the RMP. If no stock length yields a value above its cost, the current RMP solution is optimal for the full, un-enumerated LP.
- **Solving the knapsack.** A greedy fill by decreasing value/size ratio `πᵢ/ℓᵢ` usually suffices; after lengths are measured on an integer grid, the exact fallback is dynamic programming, `F_{s+1}(x) = max_{0≤r≤⌊x/ℓ_{s+1}⌋} { r π_{s+1} + F_s(x − r ℓ_{s+1}) }`, where one pass to the largest stock length prices in all stock lengths at once.
- **Integers.** A final integer master over the generated columns gives a practical cutting plan; exact integer optimality over all possible patterns requires branching with pricing.

The full pattern matrix is never formed. A tableau implementation keeps only a basis-sized matrix; a solver implementation may keep a small generated pool of columns.

## Algorithm

1. Seed with `m` homogeneous patterns — pattern `i` cuts `⌊L/ℓᵢ⌋` copies of item `i` (diagonal ⇒ feasible, invertible basis).
2. Solve the RMP LP; read duals `π`.
3. Solve the pricing knapsack `max Σ πᵢ aᵢ` s.t. `Σ ℓᵢ aᵢ ≤ L`.
4. If reduced cost `< 0` (knapsack value `> 1`), append the new pattern and go to 2; else stop — LP optimal.
5. Solve a final integer master over the generated pattern pool, or continue with branching if an exact integer certificate is required.

## Code

Uniform-stock SciPy version — LP master with dual marginals, integer knapsack pricing via `milp`, generation loop, and a final integer master over generated columns:

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

def solve_master(patterns, demand):
    A = np.asarray(patterns, dtype=float).T
    demand = np.asarray(demand, dtype=float)
    n_patterns = A.shape[1]
    res = linprog(
        np.ones(n_patterns),
        A_ub=-A,
        b_ub=-demand,
        bounds=[(0, None)] * n_patterns,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun), -res.ineqlin.marginals

def solve_pricing(stock_len, lengths, duals):
    lengths = np.asarray(lengths, dtype=float)
    ub = np.floor(stock_len / lengths)
    res = milp(
        c=-np.asarray(duals, dtype=float),
        constraints=LinearConstraint(lengths, -np.inf, stock_len),
        integrality=np.ones(len(lengths)),
        bounds=Bounds(0, ub),
    )
    if not res.success:
        raise RuntimeError(res.message)
    pattern = np.round(res.x).astype(int)
    return pattern, 1.0 - float(duals @ pattern)

def column_generation(stock_len, lengths, demand, eps=1e-6):
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

## Why it works

The pattern formulation gives a tight LP relaxation (each column is itself a feasible integer packing), unlike the roll-indexed compact model whose LP bound collapses to the area ratio `Σ ℓᵢ Nᵢ / L` and is plagued by symmetry. The cost of that tightness is exponentially many columns, but delayed column generation never pays that cost up front: it prices implicitly by solving a knapsack, touches only the few patterns the simplex method actually needs, and certifies LP optimality over all patterns when the knapsack can no longer beat the stock cost.
