import sys, json, random, isorun

# ==========================================================================
# fsx_B_0376 -- offline-decision-policy (Format B, isolated candidate)
# Theme: "asteroid mining" -- a belt refinery grades survey probes through a
# shared spectrometer with a PREFIX calibration cache. All probes are graded
# in ONE fixed global order of assay stages; each probe presents its required
# stages sorted by that order, forming a path in a prefix trie. The number of
# distinct trie nodes = calibrations actually run (nodes shared across probes
# are cache hits). Choose the global stage order to MAXIMIZE the hit-rate,
# i.e. MINIMIZE the trie node count N(order).
#
# The candidate outputs a permutation; the evaluator builds the trie ITSELF
# in this (sealed) parent process and scores. Deterministic; seeded.
# ==========================================================================

K = 3.0  # score gain per unit fractional node-reduction vs the identity order


def trie_nodes(order, probes):
    """Number of distinct prefix-trie nodes when every probe's required-stage
    subset is presented sorted by `order`. Root is not counted."""
    pos = {c: i for i, c in enumerate(order)}
    root = {}
    count = 0
    for S in probes:
        node = root
        for c in sorted(S, key=lambda x: pos[x]):
            nxt = node.get(c)
            if nxt is None:
                nxt = {}
                node[c] = nxt
                count += 1
            node = nxt
    return count


def make_instances():
    # (seed, n_stages C, n_probes Q, n_latent_groups G)
    specs = [
        (1, 8, 30, 3),
        (2, 9, 36, 4),
        (3, 10, 40, 4),
        (4, 10, 44, 4),
        (5, 12, 55, 5),
        (6, 12, 60, 6),
        (7, 13, 65, 5),
        (8, 13, 70, 6),
        (9, 14, 75, 6),
        (10, 14, 80, 7),
    ]
    out = []
    for seed, C, Q, G in specs:
        rng = random.Random(1000 + seed)
        stages = list(range(C))
        # latent mining "profiles": each group has a signature stage cluster.
        groups = []
        for _ in range(G):
            k = rng.randint(2, 4)
            groups.append(rng.sample(stages, k))
        probes = []
        for _ in range(Q):
            g = rng.randrange(G)
            S = set()
            for c in groups[g]:
                if rng.random() < 0.9:      # usually needs its group's signature
                    S.add(c)
            for _ in range(rng.randint(0, 2)):  # a little idiosyncratic noise
                S.add(rng.randrange(C))
            if not S:
                S.add(rng.randrange(C))
            probes.append(sorted(S))
        pub = {"n_stages": C, "probes": probes}
        out.append({"public": pub, "hidden": {}})
    return out


def baseline(inst):
    """Reference node count under the identity order 0,1,...,C-1."""
    pub = inst["public"]
    C = pub["n_stages"]
    return trie_nodes(list(range(C)), pub["probes"])


def _total(pub):
    return sum(len(S) for S in pub["probes"])


def score(inst, ans):
    """Validate the answer strictly, then return (ok, N) with N = trie nodes."""
    pub = inst["public"]
    C = pub["n_stages"]
    if not isinstance(ans, dict) or "order" not in ans:
        return False, 0.0
    order = ans["order"]
    if not isinstance(order, list) or len(order) != C:
        return False, 0.0
    seen = [False] * C
    clean = []
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            return False, 0.0
        if v < 0 or v >= C:
            return False, 0.0
        if seen[v]:
            return False, 0.0
        seen[v] = True
        clean.append(v)
    if not all(seen):
        return False, 0.0
    N = trie_nodes(clean, pub["probes"])
    if not isinstance(N, int) or N < 0:
        return False, 0.0
    return True, N


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, N = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        N_id = baseline(inst)
        T = _total(inst["public"])
        r = 0.1 + K * (N_id - N) / max(T, 1)
        if r != r:
            r = 0.0
        r = min(1.0, max(0.0, r))
        vec.append(r)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
