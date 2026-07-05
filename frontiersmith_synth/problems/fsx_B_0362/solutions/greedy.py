# TIER: greedy
# Keep the warehouse near-perfect, but replace the blind equal budget with a
# MARGINAL cost analysis at the lifts: starting from empty, repeatedly add one
# controller to whichever lift yields the best net cost change
# (downtime saved p*dEBO  minus  holding h_i), continuing past the economic
# optimum only as far as needed to satisfy the SLA cap.  Returns the cheaper of
# this allocation and the trivial equal-budget one.
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


def lift_marginal(s0):
    e0, _ = ebo_oh(s0, m0)
    delay = e0 / Lam if Lam > 0 else 0.0
    m = [lam[i] * (T[i] + delay) for i in range(N)]
    s = [0] * N
    ecur = [ebo_oh(0, m[i])[0] for i in range(N)]
    agg = sum(ecur)
    for _ in range(N * S_max):
        best_i = -1; best_net = 0.0
        for i in range(N):
            if s[i] >= S_max:
                continue
            enext = ebo_oh(s[i] + 1, m[i])[0]
            d = ecur[i] - enext            # backorder reduction
            net = p * d - h[i]             # positive => adding pays for itself
            if net > best_net:
                best_net = net; best_i = i
        if best_i < 0:
            break
        s[best_i] += 1
        agg -= (ecur[best_i] - ebo_oh(s[best_i], m[best_i])[0])
        ecur[best_i] = ebo_oh(s[best_i], m[best_i])[0]
    # now force feasibility (SLA) with cheapest availability per dollar
    for _ in range(N * S_max):
        if agg <= cap:
            break
        best_i = -1; best_ratio = -1.0
        for i in range(N):
            if s[i] >= S_max:
                continue
            enext = ebo_oh(s[i] + 1, m[i])[0]
            d = ecur[i] - enext
            ratio = d / h[i] if h[i] > 0 else d
            if ratio > best_ratio:
                best_ratio = ratio; best_i = i
        if best_i < 0:
            break
        agg -= (ecur[best_i] - ebo_oh(s[best_i] + 1, m[best_i])[0])
        s[best_i] += 1
        ecur[best_i] = ebo_oh(s[best_i], m[best_i])[0]
    return s, agg


# warehouse near-perfect
s0 = 0
while s0 < S0_max:
    if ebo_oh(s0, m0)[0] <= 0.02:
        break
    s0 += 1

s_marg, agg = lift_marginal(s0)

# trivial equal-budget fallback
e0, _ = ebo_oh(s0, m0); delay = e0 / Lam if Lam > 0 else 0.0
budget = cap / N * 0.9
s_triv = []
for i in range(N):
    m_i = lam[i] * (T[i] + delay); si = 0
    while si < S_max:
        if ebo_oh(si, m_i)[0] <= budget:
            break
        si += 1
    s_triv.append(si)


def total(s0v, s):
    e0, oh0 = ebo_oh(s0v, m0); delay = e0 / Lam if Lam > 0 else 0.0
    hold = h0 * oh0; a = 0.0
    for i in range(N):
        m_i = lam[i] * (T[i] + delay)
        e_i, oh_i = ebo_oh(s[i], m_i)
        hold += h[i] * oh_i; a += e_i
    return (hold + p * a) if a <= cap * (1 + 1e-9) else float("inf")


cand = [(s0, s_marg), (s0, s_triv)]
best = min(cand, key=lambda z: total(z[0], z[1]))
print(json.dumps({"s0": best[0], "s": best[1]}))
