The cutting-stock problem asks how to fill an order for many pieces of specified lengths from unlimited standard stock lengths at minimum total cost. A cutting pattern is just an integer packing of ordered pieces into one stock piece, and the exact formulation is an integer program with one variable per feasible pattern. That formulation is correct, but the set of patterns grows combinatorially with the stock width and the number of ordered lengths, so writing out the full constraint matrix is impossible for any realistic instance. The obstacle is not only integrality, which is familiar, but the sheer number of columns. The first thing to notice is that integrality can be handled after the fact: if we solve the LP relaxation, its value is a valid lower bound on the true integer cost, and in industrial instances with large demands the LP solution is usually close to integral. The real problem, then, is how to solve an LP whose columns we cannot list.

Existing ideas all fall short in a clear way. Enumerating every pattern and handing the program to a general LP or IP solver collapses as soon as the instance grows. Pure greedy heuristics that build patterns one at a time are fast but give no certificate and can waste far more trim than necessary. Kantorovich's roll-indexed compact formulation avoids enumeration, but its LP relaxation is weak — it reduces to the total-area-over-width bound and ignores the integer structure of patterns — and the symmetry among identical rolls makes the integer model very hard to solve. What is needed is the tight bound of the pattern formulation without paying the price of enumerating all patterns.

The right method is Gilmore–Gomory column generation. It starts from the pattern LP but never forms the full pattern matrix. Instead it maintains a restricted master LP over a small set of generated patterns, reads off the dual prices on the demand rows, and asks whether any pattern not yet included could improve the solution. The reduced cost of a pattern that cuts from stock of cost c is c minus the inner product of the dual prices with the pattern's piece counts. A pattern is profitable exactly when that reduced cost is negative. Rather than scanning all patterns to find one, the method searches for the best one directly.

That search is an integer knapsack problem: maximize the dual-priced value of pieces packed into one stock length, subject to the length constraint. If the maximum exceeds the stock cost, the corresponding pattern has negative reduced cost and is added to the restricted master; if not, no pattern anywhere could improve the LP, so the current restricted-master solution is optimal for the full un-enumerated LP. The subproblem is small and can be solved greedily or exactly by dynamic programming. Once the LP is optimal, a final integer master over the generated patterns produces a practical cutting plan, and the LP value gives a lower bound on its quality. Only a tiny fraction of the pattern universe is ever touched.

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
