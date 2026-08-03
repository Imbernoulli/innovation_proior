# TIER: trivial
# Full-locker gold-plate: stock every station to safety factor 4, clamped to its
# locker capacity (stock_i = min(4*sd_i, cap_i)). This is exactly the evaluator's
# baseline construction -- always feasible, heaviest holding cost -> scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]; sd = inst["sd"]; cap = inst["cap"]
stock = [min(4.0 * sd[i], cap[i]) for i in range(N)]
print(json.dumps({"stock": stock}))
