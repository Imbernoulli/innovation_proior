import sys, os, random
import numpy as np

# Resistance-portrait fitting: wire a weighted sensor mesh so that the effective
# resistance across each specified terminal pair matches a prescribed target.
#
# The target portrait is drawn from a HIDDEN feasible mesh (a random conductance
# graph), so a solver that reasons about the coupled pseudoinverse CAN approach it.
# The naive recipe -- set each pair's wire to 1/target -- cannot: effective
# resistance is far below 1/conductance because parallel paths (Rayleigh
# monotonicity) already lower it, so 1/target over-conducts and overshoots. Some
# hidden conductances exceed the exposed cap wmax, so the portrait is only PARTLY
# realizable under the constraints -> the score keeps headroom above any solution.
#
# Structure: terminals are grouped into CLUSTERS; each cluster contributes its full
# clique of coupled target pairs; consecutive cluster reps carry long-range targets.

WMAX = float(os.environ.get("GEN_WMAX", "2.5"))
ALPHA = float(os.environ.get("GEN_ALPHA", "0.85"))   # edge budget as (A-1)+ALPHA*P
TINY = float(os.environ.get("GEN_TINY", "0.50"))     # fraction of hidden conductances above the cap

LADDER = [
    (6,  3),
    (8,  3),
    (10, 3),
    (12, 4),
    (16, 4),
    (20, 5),
    (26, 5),
    (32, 6),
    (40, 6),
    (48, 7),
]


def eff_res(n, edges, pairs):
    L = np.zeros((n, n))
    for (u, v, c) in edges:
        L[u, u] += c; L[v, v] += c; L[u, v] -= c; L[v, u] -= c
    Lp = np.linalg.pinv(L, hermitian=True)
    return [float(Lp[i, i] + Lp[j, j] - 2.0 * Lp[i, j]) for (i, j) in pairs]


def main():
    tid = int(sys.argv[1])
    idx = min(max(tid, 1), len(LADDER)) - 1
    A, csz = LADDER[idx]
    rng = random.Random(1009 * tid + 71)

    verts = list(range(A))
    clusters = []
    i = 0
    while i < A:
        c = verts[i:i + csz]
        if len(c) == 1 and clusters:
            clusters[-1].extend(c)
        else:
            clusters.append(c)
        i += csz

    # specified target pairs (structure only; targets filled from hidden graph)
    pairs = []
    seen = set()

    def add_pair(u, v, w):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            return
        seen.add((a, b))
        pairs.append([a, b, w])

    for cl in clusters:
        for x in range(len(cl)):
            for y in range(x + 1, len(cl)):
                add_pair(cl[x], cl[y], round(rng.uniform(1.0, 3.0), 3))
    reps = [cl[0] for cl in clusters]
    for k in range(len(reps) - 1):
        add_pair(reps[k], reps[k + 1], round(rng.uniform(1.5, 3.5), 3))
    if len(reps) >= 3:
        for _ in range(max(1, len(reps) // 3)):
            a = rng.randrange(len(reps)); b = rng.randrange(len(reps))
            add_pair(reps[a], reps[b], round(rng.uniform(1.0, 2.5), 3))

    # ---- hidden feasible mesh: backbone path + all clique edges. A GUARANTEED
    # fraction of hidden conductances is placed ABOVE the exposed cap (at least one
    # per cluster) so the target portrait is never fully realizable -> headroom.
    hset = set()

    edge_groups = []  # list of edge-lists; above-cap forced within each
    # backbone group
    active = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    bb = []
    for k in range(len(active) - 1):
        a, b = active[k], active[k + 1]
        a, b = (a, b) if a < b else (b, a)
        if (a, b) not in hset:
            hset.add((a, b)); bb.append((a, b))
    edge_groups.append(bb)
    # one group per cluster clique
    for cl in clusters:
        ce = []
        for x in range(len(cl)):
            for y in range(x + 1, len(cl)):
                a, b = cl[x], cl[y]
                a, b = (a, b) if a < b else (b, a)
                if (a, b) not in hset:
                    hset.add((a, b)); ce.append((a, b))
        if ce:
            edge_groups.append(ce)

    hidden = []
    for grp in edge_groups:
        idxs = list(range(len(grp)))
        rng.shuffle(idxs)
        n_above = max(1, int(round(TINY * len(grp))))
        above = set(idxs[:n_above])
        for gi, (a, b) in enumerate(grp):
            if gi in above:
                g = rng.uniform(WMAX * 1.3, WMAX * 2.6)   # above the exposed cap
            else:
                g = rng.uniform(0.4, WMAX * 0.9)
            hidden.append((a, b, g))

    tp = [(p[0], p[1]) for p in pairs]
    R = eff_res(A, hidden, tp)

    P = len(pairs)
    m = (A - 1) + int(round(ALPHA * P))

    out = ["%d %d %.4f %d" % (A, m, WMAX, P)]
    for (p, r) in zip(pairs, R):
        t = max(0.05, round(r, 4))
        out.append("%d %d %.4f %.3f" % (p[0], p[1], t, p[2]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
