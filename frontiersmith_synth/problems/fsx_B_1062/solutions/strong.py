# TIER: strong
# Max-min WATER-FILLING over the relay budget instead of blind equal
# chaining. Per-hop rate log2(1+SINR) is CONCAVE in hop count (diminishing
# returns as a pair's hops shrink further), while every extra relay adds a
# co-hop transmitter that raises interference for whichever pairs are
# active in that same round. So: repeatedly hand the NEXT relay to whichever
# pair(s) currently realize the global minimum bottleneck rate (an exchange
# argument on the max-min objective), and STOP EARLY -- leaving budget
# unspent -- once no single addition can lift the current minimum. This
# naturally lands on an INTERIOR hop count per pair (favouring the
# genuinely hard/long pairs, starving already-saturated short pairs)
# instead of the greedy tier's uniform, budget-exhausting chain.
import sys, math


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_path(S, D, k):
    pts = [S]
    for j in range(1, k + 1):
        t = j / (k + 1)
        pts.append((S[0] + (D[0] - S[0]) * t, S[1] + (D[1] - S[1]) * t))
    pts.append(D)
    return pts


def per_pair_rates(pairs, alloc, P, alpha, N0):
    m = len(pairs)
    paths = [build_path(pairs[i][0], pairs[i][1], alloc[i]) for i in range(m)]
    K = max(len(pp) - 1 for pp in paths)
    min_rate = [float("inf")] * m
    for r in range(1, K + 1):
        active = []
        for i in range(m):
            if len(paths[i]) - 1 >= r:
                active.append((i, paths[i][r - 1], paths[i][r]))
        for (i, tx, rx) in active:
            d = dist(tx, rx)
            signal = P / ((1.0 + d) ** alpha)
            interf = sum(P / ((1.0 + dist(txj, rx)) ** alpha) for (j, txj, rxj) in active if j != i)
            rate = math.log2(1.0 + signal / (N0 + interf))
            if rate < min_rate[i]:
                min_rate[i] = rate
    return min_rate


def global_min(pairs, alloc, P, alpha, N0):
    return min(per_pair_rates(pairs, alloc, P, alpha, N0))


def water_fill(pairs, R, P, alpha, N0):
    m = len(pairs)
    alloc = [0] * m
    cur = global_min(pairs, alloc, P, alpha, N0)
    used = 0
    while used < R:
        pr = per_pair_rates(pairs, alloc, P, alpha, N0)
        m0 = min(pr)
        tied = sorted(range(m), key=lambda i: pr[i])
        tied = [i for i in tied if pr[i] <= m0 + 1e-9]
        cand = tied[: max(0, R - used)]
        improved = False
        if cand:
            trial = list(alloc)
            for i in cand:
                trial[i] += 1
            f2 = global_min(pairs, trial, P, alpha, N0)
            if f2 > cur + 1e-12:
                alloc, cur = trial, f2
                used += len(cand)
                improved = True
        if not improved:
            best_gain = cur
            best_p = -1
            for p in range(m):
                alloc[p] += 1
                f2 = global_min(pairs, alloc, P, alpha, N0)
                alloc[p] -= 1
                if f2 > best_gain:
                    best_gain, best_p = f2, p
            if best_p >= 0:
                alloc[best_p] += 1
                used += 1
                cur = best_gain
            else:
                break  # no relay helps the global minimum any more -> stop early
    return alloc


def main():
    t = sys.stdin.read().split()
    p = 0
    m = int(t[p]); p += 1
    R = int(t[p]); p += 1
    P = float(t[p]); p += 1
    alpha = float(t[p]); p += 1
    N0 = float(t[p]); p += 1
    p += 2  # Xmax, Ymax (unused; relays stay on-segment)
    pairs = []
    for _ in range(m):
        sx = float(t[p]); sy = float(t[p + 1]); dx = float(t[p + 2]); dy = float(t[p + 3])
        p += 4
        pairs.append(((sx, sy), (dx, dy)))

    alloc = water_fill(pairs, R, P, alpha, N0)

    out = []
    for i, (S, D) in enumerate(pairs):
        k = alloc[i]
        line = [str(k)]
        for j in range(1, k + 1):
            frac = j / (k + 1)
            x = S[0] + (D[0] - S[0]) * frac
            y = S[1] + (D[1] - S[1]) * frac
            line.append("%.6f %.6f" % (x, y))
        out.append(" ".join(line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
