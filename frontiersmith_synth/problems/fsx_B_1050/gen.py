import sys, random, math

# ---------------------------------------------------------------------------
# Instance generator for "Irreversible Merge Toward a Kept Partition".
#
# Difficulty ladder (testId 1..10):
#   N = 14 + 2*testId  (16 .. 34 points)
#   K = 3 + testId//3, capped to [3, N-6]
#   TRAP_IDS (7 of 10) chain-link the K group centers with bridge points so
#   single-linkage (nearest-pair) merging chains distinct groups together.
#   NON-TRAP tests (1, 4, 7) keep the groups spatially separated -- a control
#   regime where the naive nearest-pair heuristic is not badly punished for
#   partition choice, only for the merge-ORDER (horizon) mechanism.
#
# LAMBDA (the cost-vs-affinity trade-off weight) is SELF-SCALED per instance:
# we simulate the checker's own naive baseline construction and solve for the
# LAMBDA that makes its cumulative linkage cost consume a fixed target
# fraction of the affinity offset. This keeps the objective well-calibrated
# across the whole size ladder without hand-tuned per-test magic numbers.
# ---------------------------------------------------------------------------

TRAP_IDS = {2, 3, 5, 6, 8, 9, 10}
TARGET_COST_FRACTION = 0.71


def build_instance(test_id):
    rng = random.Random(1000 + test_id * 7919)
    N = 14 + test_id * 2
    K = 3 + (test_id // 3)
    K = min(K, N - 6)
    K = max(K, 3)
    is_trap = test_id in TRAP_IDS

    gx = int(math.sqrt(K)) + 1
    spacing = 30
    centers = [((g % gx) * spacing, (g // gx) * spacing) for g in range(K)]

    per_edge = 3
    n_bridge = 0
    edges = []
    if is_trap and K > 1:
        edges = [(g, g + 1) for g in range(K - 1)]
        n_bridge = min(len(edges) * per_edge, max(2, N // 2))

    n_regular = N - n_bridge
    sizes = [n_regular // K] * K
    for r in range(n_regular - sum(sizes)):
        sizes[r] += 1

    group_of = []
    pts = []
    for g in range(K):
        for _ in range(sizes[g]):
            jx = rng.randint(-4, 4)
            jy = rng.randint(-4, 4)
            pts.append((centers[g][0] + jx, centers[g][1] + jy))
            group_of.append(g)

    if is_trap and n_bridge > 0:
        placed = 0
        ei = 0
        while placed < n_bridge:
            gA, gB = edges[ei % len(edges)]
            cA, cB = centers[gA], centers[gB]
            slot = placed // len(edges)
            frac = (slot % per_edge + 1) / (per_edge + 1)
            bx = cA[0] + (cB[0] - cA[0]) * frac
            by = cA[1] + (cB[1] - cA[1]) * frac
            jx = rng.randint(-1, 1)
            jy = rng.randint(-1, 1)
            pts.append((int(round(bx)) + jx, int(round(by)) + jy))
            group_of.append(gA if placed % 2 == 0 else gB)
            placed += 1
            ei += 1

    # scramble identity <-> position so group membership can't be read off the index
    perm = list(range(N))
    rng.shuffle(perm)
    pts2 = [pts[perm[i]] for i in range(N)]
    group2 = [group_of[perm[i]] for i in range(N)]

    B_SAME = 14
    B_DIFF = 16
    bonus = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            noise = rng.randint(-2, 2)
            v = (B_SAME + noise) if group2[i] == group2[j] else -(B_DIFF + noise)
            bonus[i][j] = v
            bonus[j][i] = v

    ALPHA = 3.0 + 0.4 * test_id
    return N, K, ALPHA, pts2, bonus


def naive_baseline_cost(N, K, ALPHA, pts):
    """Same fixed 'queue pairing' construction the checker uses as its
    internal baseline: merge (1,2), (3,4), ... in index order, requeue the
    new cluster, repeat until K clusters remain. Returns the RAW (LAMBDA=1)
    weighted cumulative cost, so LAMBDA can be solved for directly."""
    info = {}
    for i in range(1, N + 1):
        x, y = pts[i - 1]
        info[i] = (1, float(x), float(y))
    queue = list(range(1, N + 1))
    T = N - K
    cum = 0.0
    next_id = N + 1
    t = 0
    while len(queue) > K:
        a = queue.pop(0)
        b = queue.pop(0)
        t += 1
        cntA, sxA, syA = info[a]
        cntB, sxB, syB = info[b]
        cxA, cyA = sxA / cntA, syA / cntA
        cxB, cyB = sxB / cntB, syB / cntB
        d = math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)
        raw = (cntA * cntB / (cntA + cntB)) * d
        remaining = T - t
        weight = 1.0 + ALPHA / (1.0 + remaining)
        cum += raw * weight
        new_id = next_id
        next_id += 1
        info[new_id] = (cntA + cntB, sxA + sxB, syA + syB)
        queue.append(new_id)
    return cum


def main():
    test_id = int(sys.argv[1])
    N, K, ALPHA, pts, bonus = build_instance(test_id)

    total_bonus_abs = sum(abs(bonus[i][j]) for i in range(N) for j in range(i + 1, N))
    cum = naive_baseline_cost(N, K, ALPHA, pts)
    if cum < 1e-9:
        LAMBDA = 0.0
    else:
        LAMBDA = TARGET_COST_FRACTION * total_bonus_abs / cum

    out = [f"{N} {K}", f"{ALPHA:.6f} {LAMBDA:.6f}"]
    for (x, y) in pts:
        out.append(f"{x} {y}")
    for i in range(N):
        out.append(" ".join(str(bonus[i][j]) for j in range(N)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
