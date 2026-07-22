import sys, random

# gen.py <testId> -> prints ONE instance to stdout.
#
# Instance = K sparse symmetric adjacency patterns on the SAME N vertices.
# All K patterns share ONE hidden two-level structure:
#   - a separator set S (a clique) that every pattern relies on to bridge
#     otherwise-disjoint "blocks" of vertices;
#   - B disjoint blocks, each an internal clique, each linked to S through
#     only a couple of "gateway" vertices per block (so a block can only
#     reach S -- and hence any other block -- through its gateways).
# This shared "core" skeleton is IDENTICAL, byte-for-byte, in all K patterns.
# On top of it, each pattern independently adds its OWN private noise edges,
# concentrated (hub-biased) around a handful of pattern-specific vertices so
# that, in the UNION of all K patterns, ordinary block vertices can rack up
# an apparent degree rivaling the true separator's -- while S's real role
# (bridge every block together) never shows up except through the shared
# core. Only the INTERSECTION across all K patterns filters the noise out
# and leaves the true core exposed.


def build_core(N, s_frac, block_size, gateways):
    s = max(3, int(round(N * s_frac)))
    s = min(s, N - block_size - 1)
    verts = list(range(1, N + 1))
    S = verts[:s]
    rest = verts[s:]
    blocks = [rest[i:i + block_size] for i in range(0, len(rest), block_size)]
    edges = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            edges.add((S[i], S[j]))
    for b in blocks:
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                a, c = b[i], b[j]
                edges.add((min(a, c), max(a, c)))
        gws = b[:min(gateways, len(b))]
        for g in gws:
            for sv in S:
                edges.add((min(g, sv), max(g, sv)))
    return edges, rest


def gen_patterns(N, core, noise_pool, K, noise_count, hub_frac, rng):
    patterns = []
    for _k in range(K):
        edges = set(core)
        num_hubs = max(1, int(round(len(noise_pool) * hub_frac)))
        hubs = rng.sample(noise_pool, min(num_hubs, len(noise_pool)))
        added = 0
        guard = 0
        while added < noise_count and guard < noise_count * 40 + 400:
            guard += 1
            if hubs and rng.random() < 0.7:
                u = rng.choice(hubs)
                v = rng.choice(noise_pool)
            else:
                u = rng.choice(noise_pool)
                v = rng.choice(noise_pool)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edges:
                continue
            edges.add((a, b))
            added += 1
        patterns.append(sorted(edges))
    return patterns


def main():
    tid = int(sys.argv[1])
    idx = max(1, min(10, tid)) - 1

    sizes = [40, 55, 70, 90, 110, 140, 170, 210, 250, 300]
    K = 8
    nf = 0.4

    N = sizes[idx]

    rng = random.Random(1_000_003 * tid + 17)

    core, noise_pool = build_core(N, s_frac=0.10, block_size=6, gateways=2)
    noise_count = max(1, int(round(N * nf)))
    hub_frac = 0.12

    patterns = gen_patterns(N, core, noise_pool, K, noise_count, hub_frac, rng)

    out = [f"{N} {K}"]
    for pat in patterns:
        out.append(str(len(pat)))
        out.append(" ".join(f"{u} {v}" for (u, v) in pat) if pat else "")
    print("\n".join(out))


if __name__ == "__main__":
    main()
