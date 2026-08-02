import sys, random


def build_line(n):
    return [(i, i + 1) for i in range(n - 1)]


def build_comb(K, L):
    """Spine of K nodes (0..K-1); node c also roots a pendant chain ("tooth")
    of L nodes. n = K + K*L physical qubits, n-1 edges total (a tree)."""
    n = K + K * L
    edges = [(c, c + 1) for c in range(K - 1)]
    next_id = K
    for c in range(K):
        prev = c
        for _l in range(L):
            nid = next_id
            next_id += 1
            edges.append((prev, nid))
            prev = nid
    clusters = [[c + K * t for t in range(L + 1)] for c in range(K)]
    return n, edges, clusters


CASES = {
    1: ("line", dict(n=6)),
    2: ("comb", dict(K=3, L=1)),
    3: ("line", dict(n=10)),
    4: ("comb", dict(K=4, L=2)),
    5: ("comb", dict(K=4, L=3)),
    6: ("line", dict(n=14)),
    7: ("comb", dict(K=5, L=3)),
    8: ("comb", dict(K=6, L=3)),
    9: ("comb", dict(K=7, L=3)),
    10: ("comb", dict(K=8, L=3)),
}


def plant_duplicates(rng, gates, frac=0.22):
    """Insert exact (type, unordered-pair) duplicates immediately after a
    random subset of existing gates -- guaranteed, checker-recognizable
    gate-cancellation opportunities (adjacent -> always in CANCELABLE)."""
    m0 = len(gates)
    k = max(2, int(m0 * frac))
    idxs = sorted(rng.sample(range(m0), min(k, m0)))
    out = []
    for i, g in enumerate(gates):
        out.append(g)
        if i in idxs:
            out.append(g)  # exact duplicate, consecutive -> always cancels
    return out


def main():
    testId = int(sys.argv[1])
    rng = random.Random(1_000_003 * testId + 17)

    kind, params = CASES[testId]
    if kind == "line":
        n = params["n"]
        edges = build_line(n)
        clusters = [list(range(n))]
    else:
        K, L = params["K"], params["L"]
        n, edges, _arm_clusters = build_comb(K, L)
        # decouple which LOGICAL qubits interact heavily from the physical
        # arm structure: a random (seeded) permutation groups the interaction
        # clusters, so the identity mapping (logical i on physical i) scatters
        # each cluster across multiple arms regardless of K, L parity.
        perm = list(range(n))
        rng.shuffle(perm)
        clusters = [perm[c * (L + 1):(c + 1) * (L + 1)] for c in range(K)]

    gates = []
    if kind == "line":
        m_target = n * 5
        for _ in range(m_target):
            r = rng.random()
            if r < 0.70:
                i = rng.randrange(0, n - 1)
                a, b = i, i + 1
            elif r < 0.90:
                i = rng.randrange(0, n - 1)
                j = min(n - 1, i + rng.randrange(1, 3))
                a, b = i, j
            else:
                a = rng.randrange(0, n)
                b = rng.randrange(0, n)
                while b == a:
                    b = rng.randrange(0, n)
            t = rng.randrange(0, 2)
            gates.append((t, a, b))
    else:
        seq = []
        for members in clusters:
            reps = max(3, (len(members) - 1) * 3)
            for _ in range(reps):
                a = rng.choice(members)
                b = rng.choice(members)
                while b == a:
                    b = rng.choice(members)
                t = rng.randrange(0, 2)
                seq.append((t, a, b))
        rng.shuffle(seq)
        n_cross = max(4, len(clusters))
        for _ in range(n_cross):
            a = rng.randrange(0, n)
            b = rng.randrange(0, n)
            while b == a:
                b = rng.randrange(0, n)
            t = rng.randrange(0, 2)
            seq.append((t, a, b))
        rng.shuffle(seq)
        gates = seq

    gates = plant_duplicates(rng, gates)

    out = [f"{n} {len(gates)} {len(edges)}"]
    for u, v in edges:
        out.append(f"{u} {v}")
    for t, a, b in gates:
        out.append(f"{t} {a} {b}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
