# TIER: strong
import sys

# Insight: the nominal-skew budget only requires PARTIAL closure of the raw
# wire-delay gap -- you do not have to reach zero skew. Closing the full
# gap needs many buffers; every inserted buffer's corner offset adds to
# whichever net received it, so a net that got many buffers chasing full
# balance becomes the most exposed to process corners, pushing the
# worst-case skew over its (much tighter) budget. Close only what the
# nominal budget actually demands, and do it with the lowest-variation
# buffer type -- never the "fewest insertions" type that a naive balance
# would reach for.


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1; return v
    K = int(nxt()); m = int(nxt()); C = int(nxt())
    types = []
    for _ in range(m):
        D = int(nxt()); P = int(nxt())
        Var = tuple(int(nxt()) for _ in range(C))
        types.append((D, P, Var))
    nom_budget = int(nxt()); worst_budget = int(nxt())
    ws = [int(nxt()) for _ in range(K)]

    SAFE = min(range(m), key=lambda t: max(types[t][2]) - min(types[t][2]))
    D_safe = types[SAFE][0]

    def compute(counts):
        nom = [ws[i] + sum(counts[i][t] * types[t][0] for t in range(m)) for i in range(K)]
        nom_skew = max(nom) - min(nom)
        worst = 0
        for c in range(C):
            dc = [ws[i] + sum(counts[i][t] * (types[t][0] + types[t][2][c]) for t in range(m))
                  for i in range(K)]
            worst = max(worst, max(dc) - min(dc))
        return nom, nom_skew, worst

    counts = [[0] * m for _ in range(K)]
    Tstar = max(ws)
    target = Tstar - nom_budget  # every net only needs to reach this, not Tstar
    for i in range(K):
        gap = target - ws[i]
        if gap > 0:
            counts[i][SAFE] = -(-gap // D_safe)

    nom, nom_skew, worst = compute(counts)

    # bounded safety net: if this minimal partial closure still misses
    # either budget (shouldn't happen by design, but stay robust), tighten
    # gradually using the same safe type until feasible.
    guard = 0
    while (nom_skew > nom_budget or worst > worst_budget) and guard < 6 * K + 20:
        guard += 1
        cur_nom = [ws[i] + sum(counts[i][t] * types[t][0] for t in range(m)) for i in range(K)]
        i = min(range(K), key=lambda j: cur_nom[j])
        counts[i][SAFE] += 1
        nom, nom_skew, worst = compute(counts)

    out = []
    for row in counts:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
