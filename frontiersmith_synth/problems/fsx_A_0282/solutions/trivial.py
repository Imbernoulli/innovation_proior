# TIER: trivial
# Reproduce the evaluator's weak NEXT-FIT operator: keep loading the current truck
# in arrival order; the moment a container breaks EITHER the mass or the bulk limit,
# dispatch that truck and open a fresh one.  Never look back.  This exactly matches
# q_base, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
W, V = inst["W"], inst["V"]
mass, bulk = inst["mass"], inst["bulk"]
n = len(mass)

assign = [0] * n
t = 0
rm, rb = W, V
for i in range(n):
    m, b = mass[i], bulk[i]
    if m <= rm and b <= rb:
        rm -= m; rb -= b
    else:
        t += 1
        rm = W - m; rb = V - b
    assign[i] = t

print(json.dumps({"assign": assign}))
