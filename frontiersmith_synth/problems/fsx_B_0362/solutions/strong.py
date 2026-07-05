# TIER: strong
# Co-optimize BOTH echelons.  For every candidate warehouse level s0 (from
# near-perfect down to lean), recompute the METRIC delay it induces, then run
# the same marginal cost descent + SLA top-up at the lifts, and keep the
# cheapest TOTAL (warehouse holding + lift holding + downtime) design that
# still meets the SLA cap.  This exploits warehouse risk-pooling (central
# holding h0 is cheaper than lift holding h_i) against the lead-time coupling.
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
h = inst["h"]; h0 = inst["h0"]; p = inst["p"]; cap = inst["cap"]
Lam = inst["Lambda"]; S_max = inst["S_max"]; S0_max = inst["S0_max"]
m0 = Lam * T0


def lift_alloc(delay):
    m = [lam[i] * (T[i] + delay) for i in range(N)]
    s = [0] * N
    ecur = [ebo_oh(0, m[i])[0] for i in range(N)]
    agg = sum(ecur)
    # economic marginal descent
    for _ in range(N * S_max):
        best_i = -1; best_net = 0.0
        for i in range(N):
            if s[i] >= S_max:
                continue
            enext = ebo_oh(s[i] + 1, m[i])[0]
            net = p * (ecur[i] - enext) - h[i]
            if net > best_net:
                best_net = net; best_i = i
        if best_i < 0:
            break
        s[best_i] += 1
        new = ebo_oh(s[best_i], m[best_i])[0]
        agg -= (ecur[best_i] - new); ecur[best_i] = new
    # SLA top-up: cheapest availability per dollar
    for _ in range(N * S_max):
        if agg <= cap:
            break
        best_i = -1; best_ratio = -1.0
        for i in range(N):
            if s[i] >= S_max:
                continue
            enext = ebo_oh(s[i] + 1, m[i])[0]
            ratio = (ecur[i] - enext) / h[i] if h[i] > 0 else (ecur[i] - enext)
            if ratio > best_ratio:
                best_ratio = ratio; best_i = i
        if best_i < 0:
            break
        new = ebo_oh(s[best_i] + 1, m[best_i])[0]
        agg -= (ecur[best_i] - new); s[best_i] += 1; ecur[best_i] = new
    return s, agg, m


def total(s0v, s, m):
    e0, oh0 = ebo_oh(s0v, m0)
    hold = h0 * oh0; a = 0.0
    for i in range(N):
        e_i, oh_i = ebo_oh(s[i], m[i])
        hold += h[i] * oh_i; a += e_i
    if a > cap * (1 + 1e-9):
        return float("inf")
    return hold + p * a


# near-perfect warehouse level (upper end of the sweep)
s0_hi = 0
while s0_hi < S0_max:
    if ebo_oh(s0_hi, m0)[0] <= 0.02:
        break
    s0_hi += 1

best_cost = float("inf"); best = None
for s0v in range(0, s0_hi + 1):
    e0, _ = ebo_oh(s0v, m0)
    delay = e0 / Lam if Lam > 0 else 0.0
    s, agg, m = lift_alloc(delay)
    c = total(s0v, s, m)
    if c < best_cost:
        best_cost = c; best = (s0v, s)

if best is None:
    best = (s0_hi, lift_alloc(ebo_oh(s0_hi, m0)[0] / Lam)[0])
print(json.dumps({"s0": best[0], "s": best[1]}))
