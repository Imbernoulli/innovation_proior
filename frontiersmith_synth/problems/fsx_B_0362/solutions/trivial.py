# TIER: trivial
# Decoupled equal-budget rule: stock the warehouse to near-perfect (tiny
# backorders) so the lift lead times are essentially undisturbed, then give
# EVERY lift the same expected-backorder budget cap/N.  Feasible but blind to
# per-lift economics and to warehouse pooling -> wasteful.
import sys, json, math


def ebo_oh(S, m):
    if m <= 0.0:
        return 0.0, float(max(0, S))
    S = int(S)
    xmax = S + int(m + 15.0 * math.sqrt(m) + 80.0)
    p = math.exp(-m); ebo = 0.0; x = 0
    while x <= xmax:
        if x > S:
            ebo += (x - S) * p
        x += 1
        p *= m / x
        if x > S and p < 1e-16 and (x - S) * p < 1e-18:
            break
    oh = ebo + (S - m)
    return ebo, (oh if oh > 0.0 else 0.0)


inst = json.load(sys.stdin)
N = inst["N"]; lam = inst["lam"]; T = inst["T"]; T0 = inst["T0"]
Lam = inst["Lambda"]; S_max = inst["S_max"]; S0_max = inst["S0_max"]; cap = inst["cap"]
m0 = Lam * T0
s0 = 0
while s0 < S0_max:
    e0, _ = ebo_oh(s0, m0)
    if e0 <= 0.02:
        break
    s0 += 1
e0, _ = ebo_oh(s0, m0)
delay = e0 / Lam if Lam > 0 else 0.0
budget = cap / N * 0.9
s = []
for i in range(N):
    m_i = lam[i] * (T[i] + delay)
    si = 0
    while si < S_max:
        e_i, _ = ebo_oh(si, m_i)
        if e_i <= budget:
            break
        si += 1
    s.append(si)
print(json.dumps({"s0": s0, "s": s}))
