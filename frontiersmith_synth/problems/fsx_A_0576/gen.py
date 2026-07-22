import sys, random

# hidden-block-orbit-cover / theme: courier routes through a city with secret districts
#
# We build an imprimitive permutation group G on n = b*s points (couriers' address space).
# The n points partition into b hidden DISTRICTS (blocks) of size s. Generators ("moves"):
#   move 1        : an s-cycle WITHIN every district (intra-district shuffle)  <- baseline uses this
#   moves 2..m-1  : other fixed intra-district permutations (keep each district set-wise fixed)
#   move m        : the ONLY inter-district move: sends district i -> district i+1 (mod b) with a
#                   within-district twist. So a word's district-action is (move m count) mod b.
# The district partition is HIDDEN: all points are relabelled by a secret permutation pi.
# A seed set S of t parcels lives inside district 0. Applying a word maps S to another t-subset;
# the union of images over the k chosen words is the coverage.
#
# TRAP: a random word contains move m only ~L/m times, so its images concentrate in a few
# districts near district 0 and churn WITHIN them. INSIGHT: recover the district partition
# (imprimitivity block recovery via the pair-orbit closure), find move m as the generator with a
# non-trivial district-action, and spend one short word per DISTINCT district (transversal) so
# coverage grows by ~t per word across districts instead of by ones within one district.

# per-test parameters: (b prime, s, L, k). t = (2*s)//3, m = M generators.
# The single inter-district move is RARE (1 of M) and routes are SHORT (<= L), so random routes
# almost never accumulate the many inter-district moves needed to reach far districts -- they churn
# within the seed district and its immediate neighbours. Recovering the partition lets you build the
# exact short word to each district and spend one route per DISTINCT district.
# k is set to 2L+1 (= the number of districts reachable within L on the district cycle) in build().
PARAMS = [
    (11, 12, 4),
    (13, 15, 4),
    (17, 12, 5),
    (19, 15, 5),
    (23, 12, 5),
    (23, 15, 6),
    (29, 12, 6),
    (29, 15, 6),
    (31, 18, 5),
    (37, 15, 6),
]

M = 12  # number of generators (moves): 1 inter-district (rare) + M-1 intra-district

def build(testId):
    b, s, L = PARAMS[testId - 1]
    k = 2 * L + 1
    t = (2 * s) // 3
    n = b * s
    rng = random.Random(1000 + testId)

    def pt(bl, o):
        return bl * s + o

    # generators in TRUE coordinates as image arrays g[x] = image of x
    gens = []
    # move 1: intra-district s-cycle o -> (o+1)%s (same in every district) -- baseline's move
    g = [0] * n
    for bl in range(b):
        for o in range(s):
            g[pt(bl, o)] = pt(bl, (o + 1) % s)
    gens.append(g)
    # moves 2..M-1: fixed random intra-district permutations (same across districts)
    for _ in range(M - 2):
        tau = list(range(s))
        rng.shuffle(tau)
        g = [0] * n
        for bl in range(b):
            for o in range(s):
                g[pt(bl, o)] = pt(bl, tau[o])
        gens.append(g)
    # move M: inter-district cycle bl -> (bl+1)%b with a within-district twist
    twist = list(range(s))
    rng.shuffle(twist)
    g = [0] * n
    for bl in range(b):
        for o in range(s):
            g[pt(bl, o)] = pt((bl + 1) % b, twist[o])
    gens.append(g)

    # hide the district structure: relabel all points by a secret permutation pi
    pi = list(range(n))
    rng.shuffle(pi)
    R = [0] * n  # relabelled generators
    Gens = []
    for g in gens:
        G = [0] * n
        for x in range(n):
            G[pi[x]] = pi[g[x]]
        Gens.append(G)

    # seed set S: t parcels inside district 0 (relabelled)
    S = [pi[pt(0, o)] for o in range(t)]

    return n, M, k, L, t, Gens, S

def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n, m, k, L, t, Gens, S = build(testId)
    out = []
    out.append("%d %d %d %d %d" % (n, m, k, L, t))
    for G in Gens:
        out.append(" ".join(map(str, G)))
    out.append(" ".join(map(str, S)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
