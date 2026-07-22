# TIER: greedy
# The obvious "recipe" approach: figure out which ONE solver has the best
# average requirement across the whole batch, then back that single solver
# for EVERY case (funding each fully, in case order) until the budget runs
# out. This is a real improvement over doing-nothing-clever, but it can't
# see per-case feature interactions: it ranks solvers ONCE, globally, and
# never revisits that ranking per case.
import sys, json


def req(pub, ci, j):
    p = pub["solver_profiles"][j]
    c = pub["cases"][ci]
    v = (p["base"] + p["size_coef"] * c["size"] + p["domain_coef"] * c["domain"]
         + p["inter_coef"] * c["size"] * c["domain"] + pub["case_noise"][ci][j])
    return max(pub["req_floor"], v)


inst = json.load(sys.stdin)
C, k = inst["n_cases"], inst["k"]

avg = []
for j in range(k):
    avg.append(sum(req(inst, ci, j) for ci in range(C)) / C)
jstar = min(range(k), key=lambda j: avg[j])

attempts = [[ci, jstar, req(inst, ci, jstar)] for ci in range(C)]
print(json.dumps({"attempts": attempts}))
