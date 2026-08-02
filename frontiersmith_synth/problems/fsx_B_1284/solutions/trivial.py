# TIER: trivial
# As-equal-as-possible split across eligible (non-excluded) names, water-filled against
# each name's own substitution cap so the output stays feasible. This is exactly the
# checker's internal baseline construction, so it scores ~0.1 on every case.
import sys


def equal_waterfill(target, caps):
    n = len(caps)
    alloc = [0.0] * n
    active = [True] * n
    remaining = target
    guard = 0
    while remaining > 1e-12 and guard < n + 5:
        idxs = [i for i in range(n) if active[i]]
        if not idxs:
            break
        share = remaining / len(idxs)
        newly_sat = []
        for i in idxs:
            headroom = caps[i] - alloc[i]
            if share >= headroom - 1e-12:
                alloc[i] += headroom
                remaining -= headroom
                newly_sat.append(i)
        if not newly_sat:
            for i in idxs:
                alloc[i] += share
            remaining = 0.0
            break
        for i in newly_sat:
            active[i] = False
        guard += 1
    return alloc


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    N = int(nxt())
    S = int(nxt())
    K = S + 1
    T = float(nxt())
    for _ in range(K * K):
        nxt()

    esg = []
    cap = []
    for _ in range(N):
        nxt()          # sector
        nxt()          # size
        e = float(nxt())
        nxt()          # w
        c = float(nxt())
        nxt()          # d
        esg.append(e)
        cap.append(c)

    elig_idx = [i for i in range(N) if esg[i] >= T]
    elig_caps = [cap[i] for i in elig_idx]
    elig_alloc = equal_waterfill(1.0, elig_caps)

    x = [0.0] * N
    for j, i in enumerate(elig_idx):
        x[i] = elig_alloc[j]

    out = ["%.10f" % v for v in x]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
