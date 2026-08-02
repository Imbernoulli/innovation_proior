# TIER: strong
# Insight: what the tracking-error metric charges is drift in FACTOR EXPOSURE, not drift
# in individual weights (that is the whole point of the e^T F e term). So instead of
# reweighting survivors proportionally to their own weights (greedy, factor-blind), first
# decide how much weight each SECTOR should carry, matching that sector's benchmark
# exposure as closely as substitution-availability allows -- lost weight is replaced with
# SAME-SECTOR substitutes first, rather than smeared across the whole universe.
#
# Driving every sector's exposure gap to exactly zero is optimal for ANY positive
# semi-definite F (e^T F e = 0 whenever e = 0), so matching capacity allows is always the
# first priority. The only place F's actual numbers matter is when substitution capacity
# is SCARCE and a shortfall has to be rationed across sectors that cannot be fully
# restored: then the sector whose mismatch the covariance matrix charges the most (its own
# variance F[k][k], PLUS how it co-moves with the size factor) should get first claim on
# whatever spare capacity remains, ahead of sectors F prices more cheaply.
import sys


def waterfill(target, items):
    """items: list of (key, desired_weight, cap). Allocate up to `target` total, each key
    capped at its `cap`, proportional to desired_weight among unsaturated keys, spillover
    from saturated keys redistributed among the rest. Returns {key: alloc}."""
    alloc = {}
    active = [it for it in items if it[2] > 1e-15]
    for k, _, _ in items:
        alloc.setdefault(k, 0.0)
    remaining = target
    guard = 0
    while active and remaining > 1e-12 and guard < len(items) + 5:
        wsum = sum(wt for _, wt, _ in active)
        if wsum <= 1e-15:
            share = remaining / len(active)
            for k, _, cap in active:
                a = min(cap, share)
                alloc[k] += a
                remaining -= a
            break
        scale = remaining / wsum
        saturated = []
        for k, wt, cap in active:
            headroom = cap - alloc[k]
            if wt * scale >= headroom - 1e-12:
                alloc[k] += headroom
                remaining -= headroom
                saturated.append(k)
        if not saturated:
            for k, wt, cap in active:
                alloc[k] += wt * scale
            remaining = 0.0
            break
        active = [it for it in active if it[0] not in saturated]
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
    F = [[float(nxt()) for _ in range(K)] for _ in range(K)]

    sectors, sizes, esg, w, cap, d = [], [], [], [], [], []
    for _ in range(N):
        sectors.append(int(nxt()))
        sizes.append(float(nxt()))
        esg.append(float(nxt()))
        w.append(float(nxt()))
        cap.append(float(nxt()))
        d.append(float(nxt()))

    eligible = [e >= T for e in esg]

    # benchmark sector exposure targets (over ALL names, including excluded -- that is
    # what the screened portfolio is still trying to track)
    sector_target = [0.0] * S
    for i in range(N):
        sector_target[sectors[i]] += w[i]

    sector_capacity = [0.0] * S
    for i in range(N):
        if eligible[i]:
            sector_capacity[sectors[i]] += cap[i]

    # phase A: give every sector as much of its own target as capacity allows -- this
    # alone drives that sector's factor-exposure gap towards 0, which is optimal for any
    # PSD F, independent of F's actual numbers.
    sector_alloc = [min(sector_target[k], sector_capacity[k]) for k in range(S)]

    # phase B: whatever shortfall remains (some sector's capacity couldn't cover its own
    # target) is rationed across sectors that still have spare headroom, weighted by how
    # expensive THIS covariance matrix charges that sector's mismatch: its own variance
    # F[k][k] plus its co-movement with the size factor F[k][K-1] (risk-priority instead
    # of blind proportional spillover).
    shortfall = 1.0 - sum(sector_alloc)
    if shortfall > 1e-12:
        priced = []
        for k in range(S):
            headroom = sector_capacity[k] - sector_alloc[k]
            if headroom > 1e-12:
                risk = F[k][k] + abs(F[k][K - 1])
                priced.append((k, max(risk, 1e-9), headroom))
        got = waterfill(shortfall, priced)
        for k, _, _ in priced:
            sector_alloc[k] += got[k]

    x = [0.0] * N
    for k in range(S):
        names_k = [(i, w[i], cap[i]) for i in range(N) if eligible[i] and sectors[i] == k]
        if not names_k:
            continue
        got = waterfill(sector_alloc[k], names_k)
        for i, _, _ in names_k:
            x[i] = got[i]

    # numerical cleanup: any residual (from waterfill guard limits, or leftover capacity
    # nobody claimed because total sector targets summed to < 1) goes to whichever
    # eligible name still has the most spare capacity, keeping the output feasible
    resid = 1.0 - sum(x)
    if abs(resid) > 1e-9:
        cand = sorted((i for i in range(N) if eligible[i]), key=lambda i: cap[i] - x[i], reverse=True)
        for i in cand:
            room = cap[i] - x[i] if resid > 0 else x[i]
            take = min(abs(resid), room) if resid > 0 else -min(abs(resid), room)
            x[i] += take
            resid -= take
            if abs(resid) <= 1e-9:
                break

    out = ["%.10f" % v for v in x]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
