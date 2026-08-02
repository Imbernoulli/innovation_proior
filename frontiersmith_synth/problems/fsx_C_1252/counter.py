import sys

# counter.py <in> <out> <ans>  -- deterministic scorer for clock-tree balance.
# Reads the instance (sink wire delays, buffer library, budgets), the
# participant's buffer-count plan, validates feasibility strictly, then
# scores total buffer power against the checker's own defensive baseline.
# Prints "... Ratio: <float>".

TYPES = [
    (6,  5, (-1, 0, 1, 1)),
    (9,  4, (-3, 1, 2, 4)),
    (16, 6, (-7, 2, 3, 11)),
]
NTYPES = len(TYPES)
NCORNERS = 4


def fail(msg):
    print("INFEASIBLE: %s Ratio: 0.0" % msg)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = toks[p]; p += 1; return v
    K = int(nxt()); m = int(nxt()); C = int(nxt())
    assert m == NTYPES and C == NCORNERS, "instance/library mismatch"
    types = []
    for _ in range(m):
        D = int(nxt()); P = int(nxt())
        Var = tuple(int(nxt()) for _ in range(C))
        types.append((D, P, Var))
    nom_budget = int(nxt()); worst_budget = int(nxt())
    ws = [int(nxt()) for _ in range(K)]
    return K, m, C, types, nom_budget, worst_budget, ws


def parse_int_token(tok):
    if tok is None:
        return None
    s = tok
    neg = False
    if s.startswith('+'):
        s = s[1:]
    elif s.startswith('-'):
        neg = True
        s = s[1:]
    if s == '' or not s.isdigit():
        return None
    v = int(s)
    return -v if neg else v


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    K, m, C, types, nom_budget, worst_budget, ws = read_instance(in_path)

    with open(out_path) as f:
        raw = f.read().split()

    if len(raw) != K * m:
        fail("expected %d tokens (K*m), got %d" % (K * m, len(raw)))

    counts = []
    for i in range(K):
        row = []
        for t in range(m):
            tok = raw[i * m + t]
            v = parse_int_token(tok)
            if v is None:
                fail("non-integer / non-finite token at sink %d type %d: %r" % (i, t, tok))
            if v < 0:
                fail("negative buffer count at sink %d type %d" % (i, t))
            row.append(v)
        counts.append(row)

    # nominal (corner-free) skew
    nom = [ws[i] + sum(counts[i][t] * types[t][0] for t in range(m)) for i in range(K)]
    nom_skew = max(nom) - min(nom)
    if nom_skew > nom_budget:
        fail("nominal skew %d exceeds budget %d" % (nom_skew, nom_budget))

    # worst-case skew across deterministic process-variation corners
    worst_skew = 0
    for c in range(C):
        dc = [ws[i] + sum(counts[i][t] * (types[t][0] + types[t][2][c]) for t in range(m))
              for i in range(K)]
        worst_skew = max(worst_skew, max(dc) - min(dc))
    if worst_skew > worst_budget:
        fail("worst-case skew %d exceeds budget %d" % (worst_skew, worst_budget))

    # objective: total buffer power (fewer/cheaper buffers = better)
    F = sum(counts[i][t] * types[t][1] for i in range(K) for t in range(m))

    # checker's own reference plan: type 0 (safe, low variation) only,
    # buffers added to every net until its nominal delay reaches the
    # instance maximum. Always feasible by construction (both budgets in
    # the input were derived from exactly this plan plus positive slack in
    # gen.py) but not power-efficient.
    D0, P0 = types[0][0], types[0][1]
    Tstar = max(ws)
    base_counts = [[0] * m for _ in range(K)]
    for i in range(K):
        gap = Tstar - ws[i]
        if gap > 0:
            base_counts[i][0] = -(-gap // D0)
    B = sum(base_counts[i][t] * types[t][1] for i in range(K) for t in range(m))
    B = max(B, 1)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("OK power=%d baseline=%d nom_skew=%d/%d worst_skew=%d/%d Ratio: %.6f"
          % (F, B, nom_skew, nom_budget, worst_skew, worst_budget, sc / 1000.0))


if __name__ == "__main__":
    main()
