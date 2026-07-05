# TIER: strong
# Multi-rule seeding + deterministic insertion local search over the priority order,
# scored on the SAME non-delay simulator the yardmaster uses.  It seeds from several
# classic dispatch rules -- EDD (earliest cut-off), WSPT (weight/hump-time), FCFS, and
# two Apparent-Tardiness-Cost (ATC) rules with different look-ahead scales -- keeps the
# best, then repeatedly relocates a single cut to the position that most reduces total
# weighted lateness until no move helps.  This weaves together due-dates, weights and
# release staggering, beating any single static rule; but 1|r_j|sum w_j T_j is NP-hard
# and the search is a bounded local optimum, so it stays well short of the ideal.
import sys, json, math

inst = json.load(sys.stdin)
cuts = inst["cuts"]
n = len(cuts)
P = [c["p"] for c in cuts]
W = [c["w"] for c in cuts]
R = [c["r"] for c in cuts]
D = [c["d"] for c in cuts]


def simulate(order):
    pos = [0] * n
    for k, j in enumerate(order):
        pos[j] = k
    done = [False] * n
    t = 0
    total = 0.0
    ndone = 0
    while ndone < n:
        best = -1
        best_pos = 1 << 30
        next_r = 1 << 60
        for j in range(n):
            if done[j]:
                continue
            if R[j] <= t:
                if pos[j] < best_pos:
                    best_pos = pos[j]
                    best = j
            elif R[j] < next_r:
                next_r = R[j]
        if best < 0:
            t = next_r
            continue
        t += P[best]
        late = t - D[best]
        if late > 0:
            total += W[best] * late
        done[best] = True
        ndone += 1
    return total


def edd():
    return sorted(range(n), key=lambda i: (D[i], i))


def wspt():
    return sorted(range(n), key=lambda i: W[i] / P[i], reverse=True)


def fcfs():
    return sorted(range(n), key=lambda i: (R[i], i))


def atc(K):
    pbar = sum(P) / n
    def key(i):
        slack = D[i] - P[i] - R[i]
        if slack < 0:
            slack = 0
        return -(W[i] / P[i]) * math.exp(-slack / (K * pbar + 1e-9))
    return sorted(range(n), key=key)


seeds = [edd(), wspt(), fcfs(), atc(1.0), atc(3.0)]
best_val = None
best_order = None
for s in seeds:
    v = simulate(s)
    if best_val is None or v < best_val:
        best_val = v
        best_order = s[:]

seq = best_order[:]
improved = True
rounds = 0
while improved and rounds < 200:
    improved = False
    rounds += 1
    for a in range(n):
        x = seq[a]
        rest = seq[:a] + seq[a + 1:]
        for b in range(n):
            cand = rest[:b] + [x] + rest[b:]
            v = simulate(cand)
            if v < best_val - 1e-9:
                best_val = v
                seq = cand
                improved = True
                break
        if improved:
            break

print(json.dumps({"order": seq}))
