import sys, random

# ---- dependency-resolution-solve instance generator ------------------------
# n packages, global indices 0..n-1 IN A FIXED DECLARATION ORDER:
#   [ A_0 .. A_{K-1} ]  [ B_0 .. B_{K-1} ]  [ Z_0 .. Z_{K-1} ]  [ filler_0 .. ]
# Each package i has versions 1..m_i. Version v of package i may carry
# REQUIREMENT edges (target j, lo, hi) meaning: "if i is installed at
# version v, package j's installed version must lie in [lo,hi]". In this
# construction only A_i and B_i carry requirements, and both target the
# SAME shared package Z_i (a diamond: A_i -> Z_i <- B_i). Note j > i always
# (a package can only require something declared LATER), so a one-pass
# resolver that walks packages 0..n-1 committing to a version as it goes
# cannot verify A_i's or B_i's requirement until it reaches Z_i, many
# packages later.
#
# EASY gadgets: newest-A + newest-B always leaves Z_i a non-empty feasible
# range (first-try success), so no backtracking is ever needed for them.
#
# BROKEN gadget (gadget index 0, planted only on TRAP testIds): A_0's
# NEWEST version requires Z_0 in a narrow high range that NO version of B_0
# can ever reach -> newest A_0 is a global dead end. Because the declaration
# order puts every OTHER gadget's B and Z packages between "the point A_0's
# mistake becomes visible" (at Z_0, the very start of the Z-phase) and A_0
# itself (the very first package), a one-pass chronological resolver that
# always retries the newest untried version at the MOST RECENT decision
# point is forced to re-permute every later B_i and A_i (all irrelevant to
# gadget 0) before it ever reconsiders A_0 -- a combinatorial explosion in
# the number of gadgets K, even though the true fix (drop A_0 one version)
# is a single local change. version==1 for every package is ALWAYS globally
# feasible (the universal fallback / checker baseline).

MVER = 4     # A/B version count per gadget
MZ = 6       # Z version count per gadget
MFILL = 3    # filler package version count

# testId -> (K gadgets, F fillers, broken?)
SCALE = {
    1: (2, 2, False),
    2: (3, 3, False),
    3: (4, 3, False),
    4: (4, 3, True),
    5: (3, 4, False),
    6: (5, 4, True),
    7: (6, 4, True),
    8: (5, 5, False),
    9: (7, 5, True),
    10: (8, 6, True),
}


def pref_table(rng, m):
    base = rng.randint(2, 10)
    trend = rng.randint(4, 9)
    vals = []
    for v in range(1, m + 1):
        noise = rng.randint(-6, 6)
        p = base + trend * (v - 1) + noise
        vals.append(max(1, min(100, p)))
    return vals


def easy_ranges(rng):
    """newest-A/newest-B always intersect; version1/version1 always intersect."""
    off = rng.randint(0, 1)
    RA = {av: (1, min(MZ, av + 2 + off)) for av in range(1, MVER + 1)}
    RB = {bv: (max(1, bv - off), MZ) for bv in range(1, MVER + 1)}
    return RA, RB


def broken_ranges():
    """A's newest version (4) demands Z in [5,6]; every B version only ever
    opens up Z in [1,4] -- newest A is unreachable no matter what B is.
    Versions 1..3 of A stay compatible with every B (an escape exists)."""
    RA = {av: (1, min(MZ, av + 2)) for av in range(1, MVER)}
    RA[MVER] = (5, 6)
    RB = {bv: (1, 4) for bv in range(1, MVER + 1)}
    return RA, RB


def build(t, rng):
    K, F, broken = SCALE[t]
    n = 3 * K + F
    versions = [0] * n
    prefs = [None] * n
    reqs = [None] * n  # reqs[i][v] = list of (target, lo, hi)

    def idxA(i): return i
    def idxB(i): return K + i
    def idxZ(i): return 2 * K + i

    for i in range(K):
        ai, bi, zi = idxA(i), idxB(i), idxZ(i)
        versions[ai], versions[bi], versions[zi] = MVER, MVER, MZ
        prefs[ai] = pref_table(rng, MVER)
        prefs[bi] = pref_table(rng, MVER)
        prefs[zi] = pref_table(rng, MZ)
        if broken and i == 0:
            RA, RB = broken_ranges()
        else:
            RA, RB = easy_ranges(rng)
        reqs[ai] = {av: [(zi,) + RA[av]] for av in range(1, MVER + 1)}
        reqs[bi] = {bv: [(zi,) + RB[bv]] for bv in range(1, MVER + 1)}
        reqs[zi] = {v: [] for v in range(1, MZ + 1)}

    for f in range(F):
        idx = 3 * K + f
        versions[idx] = MFILL
        prefs[idx] = pref_table(rng, MFILL)
        reqs[idx] = {v: [] for v in range(1, MFILL + 1)}

    return n, versions, prefs, reqs


def main():
    t = int(sys.argv[1])
    rng = random.Random(20260801 + 131 * t)
    n, versions, prefs, reqs = build(t, rng)

    lines = [str(n)]
    for i in range(n):
        m = versions[i]
        lines.append(str(m))
        for v in range(1, m + 1):
            edges = reqs[i][v]
            parts = [str(prefs[i][v - 1]), str(len(edges))]
            for (j, lo, hi) in edges:
                parts += [str(j), str(lo), str(hi)]
            lines.append(" ".join(parts))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
