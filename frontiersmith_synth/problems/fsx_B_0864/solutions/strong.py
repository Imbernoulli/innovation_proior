# TIER: strong
# The insight: the best EXPECTED (average) solver is not the best solver
# PER INSTANCE -- read the per-case feature interaction and re-decide the
# solver choice for every single case. For each case, evaluate all k
# solvers' requirements and keep only the cheapest (case, solver, cost).
# Then -- an exchange argument -- since every case is worth the same
# 1/n_cases and costs are additive against one shared budget, the schedule
# that maximizes the COUNT of cases solved is to spend on the cheapest
# cases FIRST: sort by that per-case minimal cost ascending and fund them
# in that order until the budget is exhausted. This both exploits the
# planted feature interaction (per-case solver choice) AND uses the shared
# budget efficiently (ascending-cost order), instead of a single global
# ranking spent blindly on every case.
import sys, json


def req(pub, ci, j):
    p = pub["solver_profiles"][j]
    c = pub["cases"][ci]
    v = (p["base"] + p["size_coef"] * c["size"] + p["domain_coef"] * c["domain"]
         + p["inter_coef"] * c["size"] * c["domain"] + pub["case_noise"][ci][j])
    return max(pub["req_floor"], v)


inst = json.load(sys.stdin)
C, k = inst["n_cases"], inst["k"]

costs = []
for ci in range(C):
    best_j, best_c = 0, None
    for j in range(k):
        r = req(inst, ci, j)
        if best_c is None or r < best_c:
            best_c, best_j = r, j
    costs.append((best_c, ci, best_j))
costs.sort(key=lambda t: (t[0], t[1]))

attempts = [[ci, j, c] for (c, ci, j) in costs]
print(json.dumps({"attempts": attempts}))
