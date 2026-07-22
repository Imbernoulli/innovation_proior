import sys, random

# ---- fixed physical constants (also embedded, unscaled, in the input for the solver) ----
DUR = 8            # every station's visibility-window duration, in ticks
GAP = 10           # spacing between consecutive stations' window offsets

def station_params(K):
    # station 0 is always the "best" station (highest drain) -- a deliberate lure.
    drains = [16] + [8] * (K - 1)
    offsets = [k * GAP for k in range(K)]
    durs = [DUR] * K
    return drains, offsets, durs

def spread_phases(rng, ids, margin):
    n = len(ids)
    phases = {}
    if n == 0:
        return phases
    step = margin / n
    order = list(ids)
    rng.shuffle(order)
    for pos, i in enumerate(order):
        phases[i] = int((pos * step) % margin)
    return phases

def make_phases(rng, S, margin, cluster_frac, jitter, n_clusters):
    idx = list(range(S))
    rng.shuffle(idx)
    clustered_n = int(round(S * cluster_frac))
    clustered_n = min(S, max(0, clustered_n))
    phases = [0] * S
    if clustered_n == 0 or n_clusters == 0:
        rest_phases = spread_phases(rng, list(range(S)), margin)
        for i in range(S):
            phases[i] = rest_phases[i]
        return phases

    clustered_ids = idx[:clustered_n]
    rest_ids = idx[clustered_n:]

    # split clustered satellites across n_clusters cluster centers
    groups = [[] for _ in range(n_clusters)]
    for pos, i in enumerate(clustered_ids):
        groups[pos % n_clusters].append(i)

    usable = max(1, margin - jitter)
    centers = []
    for g in range(n_clusters):
        centers.append(int((g + 0.5) * usable / n_clusters))

    for g in range(n_clusters):
        for i in groups[g]:
            phases[i] = min(margin - 1, centers[g] + rng.randrange(0, jitter + 1))

    rest_phases = spread_phases(rng, rest_ids, margin)
    for i in rest_ids:
        phases[i] = rest_phases[i]
    return phases

def build_case(testId):
    rng = random.Random(20260718 + 97 * testId)

    ladder = {
        1:  dict(S=3,  K=2, R=4,  cluster_frac=0.0,  jitter=0, n_clusters=0, cap_factor=20),
        2:  dict(S=4,  K=2, R=5,  cluster_frac=0.0,  jitter=0, n_clusters=0, cap_factor=20),
        3:  dict(S=5,  K=2, R=6,  cluster_frac=0.4,  jitter=1, n_clusters=1, cap_factor=18),
        4:  dict(S=6,  K=3, R=8,  cluster_frac=0.5,  jitter=1, n_clusters=1, cap_factor=17),
        5:  dict(S=8,  K=3, R=10, cluster_frac=0.5,  jitter=2, n_clusters=1, cap_factor=16),
        6:  dict(S=10, K=3, R=12, cluster_frac=0.6,  jitter=1, n_clusters=1, cap_factor=13),
        7:  dict(S=11, K=3, R=14, cluster_frac=0.65, jitter=2, n_clusters=1, cap_factor=12),
        8:  dict(S=13, K=4, R=16, cluster_frac=0.7,  jitter=1, n_clusters=2, cap_factor=12),
        9:  dict(S=14, K=4, R=18, cluster_frac=0.7,  jitter=2, n_clusters=2, cap_factor=11),
        10: dict(S=16, K=4, R=20, cluster_frac=0.75, jitter=1, n_clusters=2, cap_factor=11),
    }
    p = ladder[testId]
    S, K, R = p["S"], p["K"], p["R"]
    # MARGIN scales with S so satellites NOT deliberately clustered are always
    # spaced further apart than a window duration -- accidental collisions must
    # never happen; only the planted clusters collide.
    MARGIN = S * (DUR + 2)
    P = MARGIN + K * GAP
    T = P * R

    drains, offsets, durs = station_params(K)
    phases = make_phases(rng, S, MARGIN, p["cluster_frac"], p["jitter"], p["n_clusters"])

    sats = []
    for i in range(S):
        acc = rng.randint(2, 4)
        cap = acc * p["cap_factor"]
        sats.append((acc, cap, phases[i]))

    lines = [f"{S} {K} {P} {R} {T}"]
    for acc, cap, ph in sats:
        lines.append(f"{acc} {cap} {ph}")
    for k in range(K):
        lines.append(f"{drains[k]} {offsets[k]} {durs[k]}")
    return "\n".join(lines) + "\n"

def main():
    testId = int(sys.argv[1])
    sys.stdout.write(build_case(testId))

if __name__ == "__main__":
    main()
