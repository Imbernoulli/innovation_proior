# TIER: strong
# Decreasing-order vector packing.  We try several sort keys (each captures a
# different notion of "big" for a 2-vector) and several fit policies, then keep
# whichever plan dispatches the fewest trucks.  Seating the bulky/heavy containers
# first lets small ones top off partial trucks on both axes, so waste falls well
# below the online rules -- yet the loose L1 vector bound keeps scores under 1.0.
import sys, json

inst = json.load(sys.stdin)
W, V = inst["W"], inst["V"]
mass, bulk = inst["mass"], inst["bulk"]
n = len(mass)


def keys():
    # normalized-sum, max-axis, and per-axis dominant orderings
    yield sorted(range(n), key=lambda i: mass[i] / W + bulk[i] / V, reverse=True)
    yield sorted(range(n), key=lambda i: max(mass[i] / W, bulk[i] / V), reverse=True)
    yield sorted(range(n), key=lambda i: mass[i], reverse=True)
    yield sorted(range(n), key=lambda i: bulk[i], reverse=True)


def pack(order, policy):
    rem_m, rem_b = [], []
    assign = [0] * n
    for i in order:
        m, b = mass[i], bulk[i]
        pick, best = -1, None
        for t in range(len(rem_m)):
            if rem_m[t] >= m and rem_b[t] >= b:
                if policy == "first":
                    pick = t
                    break
                # best-fit: tightest residual on the normalized 2-vector
                score = (rem_m[t] - m) / W + (rem_b[t] - b) / V
                if best is None or score < best:
                    best = score; pick = t
        if pick < 0:
            rem_m.append(W - m); rem_b.append(V - b)
            assign[i] = len(rem_m) - 1
        else:
            rem_m[pick] -= m; rem_b[pick] -= b
            assign[i] = pick
    return assign, len(rem_m)


best_assign, best_k = None, None
for order in keys():
    for policy in ("first", "best"):
        a, k = pack(order, policy)
        if best_k is None or k < best_k:
            best_k = k; best_assign = a

print(json.dumps({"assign": best_assign}))
