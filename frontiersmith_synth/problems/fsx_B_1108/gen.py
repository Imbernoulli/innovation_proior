import sys, random

# ---------------------------------------------------------------------------
# radiative-hotspot-sinks  (format C, minimize steady-state hotspot T)
#   `python3 gen.py <testId>`  prints ONE instance to stdout.
#   Deterministic in testId only.
#
# Instance:
#   line 1:  n m k
#   line 2:  g_0 .. g_{n-1}        heat generation per junction
#   line 3:  a_0 .. a_{n-1}        radiative coefficient per junction
#   next m lines:  u v c           undirected conductive edges
#
# Layout (node indices in blocks):
#   [shell]  low g, high a  -> always cool; the trivial baseline (sinks 0..k-1)
#            wastes its budget here.
#   [filler] medium g/a, well connected bulk.
#   [core]   HIGH g, low a, dense internal straps, weak coupling to filler:
#            the loud bait.  With no sinks it is the hottest region.
#   [pockets] (trap cases) MODERATE g, very low a, connected to the rest by a
#            single thin neck: heat cannot conduct away and T^4 radiation is
#            too weak at moderate T, so they sit just below the uncooled core.
#            Once the core is sunk they become the hotspot -- and generation-
#            ranked greedy never touches them because their g is unremarkable.
#   Honest cases give the pockets the HIGHEST generation, so the generation
#   recipe accidentally lands on them (trap disarmed).
# ---------------------------------------------------------------------------

TRAP_IDS = {2, 3, 5, 6, 8, 9, 10}


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    rng = random.Random(60221023 + 7919 * t)
    trap = t in TRAP_IDS

    k = 4 + (t - 1) // 3                 # 4,4,4,5,5,5,6,6,6,7
    n_pockets = max(2, k - 2)
    psize = 3 + (t % 3)                  # 4,5,3,4,5,3,...
    if not trap:
        n_pockets = 2
        psize = 3
    n_core = k + 1
    n_shell = k + 2
    n_fill = 40 + 25 * (t - 1)

    n = n_shell + n_fill + n_core + n_pockets * psize

    sh0 = 0
    fi0 = sh0 + n_shell
    co0 = fi0 + n_fill
    po0 = co0 + n_core

    def pocket_node(p, j):
        return po0 + p * psize + j

    # ---- per-node generation / radiation -----------------------------------
    g = [0.0] * n
    a = [0.0] * n
    for i in range(sh0, fi0):            # shell: feeble sources, strong radiators
        g[i] = rng.uniform(0.2, 0.6)
        a[i] = rng.uniform(0.04, 0.09)
    for i in range(fi0, co0):            # filler bulk
        g[i] = rng.uniform(1.0, 2.6)
        a[i] = rng.uniform(0.02, 0.05)
    for i in range(co0, po0):            # core: loud, poorly radiating bait
        g[i] = rng.uniform(8.5, 10.0)
        a[i] = rng.uniform(0.0005, 0.0009)
    for p in range(n_pockets):
        for j in range(psize):
            i = pocket_node(p, j)
            if trap:
                g[i] = rng.uniform(3.4, 4.2)     # modest source ...
                a[i] = rng.uniform(0.0007, 0.0011)  # ... but barely radiates
            else:
                g[i] = rng.uniform(9.0, 11.0)    # honest: loudest of all
                a[i] = rng.uniform(0.0012, 0.0020)

    # ---- edges --------------------------------------------------------------
    edges = {}

    def block(u):
        if u < fi0:
            return "shell"
        if u < co0:
            return "fill"
        if u < po0:
            return "core"
        return "pocket"

    def add(u, v, c):
        if u == v:
            return
        bu, bv = block(u), block(v)
        if bu != bv:
            # any cross-block strap is a THIN neck: bottlenecks are the whole
            # point, so no random spanning edge may short-circuit them.
            if "core" in (bu, bv):
                c = rng.uniform(0.10, 0.35)
            elif "pocket" in (bu, bv):
                c = rng.uniform(0.08, 0.30) if trap else rng.uniform(0.30, 0.80)
            elif "shell" in (bu, bv):
                c = rng.uniform(0.8, 2.5)
        e = (u, v) if u < v else (v, u)
        if e not in edges:
            edges[e] = c

    # spanning chain for connectivity
    perm = list(range(n))
    rng.shuffle(perm)
    for w in range(1, n):
        u = perm[w]
        v = perm[rng.randrange(w)]
        add(u, v, rng.uniform(0.6, 2.0))

    # filler bulk: well connected
    for _ in range(int(1.8 * n_fill)):
        u = rng.randrange(fi0, co0)
        v = rng.randrange(fi0, co0)
        add(u, v, rng.uniform(0.8, 3.0))
    # shell hangs off filler
    for i in range(sh0, fi0):
        for _ in range(2):
            add(i, rng.randrange(fi0, co0), rng.uniform(0.8, 2.5))
    # core: complete internal straps (heat sloshes freely inside the core)
    for u in range(co0, po0):
        for v in range(u + 1, po0):
            add(u, v, rng.uniform(2.5, 5.0))
    # core to filler: thin drains (core stays hot when unsunk)
    for u in range(co0, po0):
        add(u, rng.randrange(fi0, co0), rng.uniform(0.15, 0.45))
    # pockets: complete inside (one sink chills the whole huddle)
    for p in range(n_pockets):
        for u_ in range(psize):
            for v_ in range(u_ + 1, psize):
                add(pocket_node(p, u_), pocket_node(p, v_), rng.uniform(2.5, 5.0))
    # necks: pocket <-> filler, deliberately thin
    for p in range(n_pockets):
        j = rng.randrange(psize)
        if trap:
            add(pocket_node(p, j), rng.randrange(fi0, co0), rng.uniform(0.08, 0.30))
        else:
            add(pocket_node(p, j), rng.randrange(fi0, co0), rng.uniform(0.3, 0.8))
            j2 = (j + 1) % psize
            add(pocket_node(p, j2), rng.randrange(fi0, co0), rng.uniform(0.3, 0.8))

    elist = sorted(edges.items())        # deterministic order
    m = len(elist)

    out = []
    out.append("%d %d %d" % (n, m, k))
    out.append(" ".join("%.6f" % x for x in g))
    out.append(" ".join("%.6f" % x for x in a))
    for (u, v), c in elist:
        out.append("%d %d %.6f" % (u, v, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
