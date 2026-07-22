#!/usr/bin/env python3
"""gen.py <testId> -- shared-grade-allocation instance generator.

Instance format (stdout):
  line 1: "m C"                     -- m features (0..m-1), C requirement chains
  next C lines: "p1 p2 h spec_primary spec_backup"
      p1, p2  -- indices of two CHAIN-PRIVATE features (used by this chain only)
      h       -- index of a HUB feature (may be shared by many chains)
      spec_primary  -- PRIMARY check bound: tol(p1)+tol(p2)+tol(h) <= spec_primary
      spec_backup   -- BACKUP (worst-case fallback) check bound: tol(p1)+tol(p2) <= spec_backup
                       (must hold even when the shared hub is unavailable/degraded)

Grades g in [0,6]; cost(g) = 2^g - 1; tol(g) = 2^(6-g).  Both tables are fixed and are
restated in statement.md.  Feature indices: privates are 0 .. 2C-1 (chain i owns
2i, 2i+1); hub-pool features are 2C .. 2C+Hn-1.

Deterministic: all randomness seeded purely from testId.
"""
import random
import sys

TOL = [64, 32, 16, 8, 4, 2, 1]
COST = [0, 1, 3, 7, 15, 31, 63]

# (sp_lo, sp_hi, sb_lo, sb_hi) bands.  Each band was chosen offline so that, for a
# chain drawn from the band, the cost-minimal grade for its hub member IN ISOLATION
# (gh_local) is a STRICT local optimum: moving the hub to a higher grade gh_high
# costs strictly more than it saves for a single chain, but saves enough that a
# cluster of >= a handful of chains sharing that hub strictly prefers gh_high.
BANDS = [
    (28, 31, 24, 34),   # gh_local=3 -> gh_high=4 is worth it once >=~2 chains share the hub
    (26, 27, 24, 40),   # gh_local=3 -> gh_high=5 is worth it once >=~6 chains share the hub
]

# broad, unstructured band used for "solo"/small-degree chains (adds heterogeneity;
# for these chains the backup check binds at least as often as the primary one)
SOLO_SP = (20, 52)
SOLO_SB = (10, 40)


def build(rnd, cluster_sizes, cluster_bands, n_solo, solo_hub_pool):
    """Return (chains, Hn). chains[i] = (hub_index, spec_primary, spec_backup)."""
    chains = []
    hub_of = []
    n_cluster_hubs = len(cluster_sizes)
    for hub_id, (size, band) in enumerate(zip(cluster_sizes, cluster_bands)):
        sp_lo, sp_hi, sb_lo, sb_hi = band
        for _ in range(size):
            sp = rnd.randint(sp_lo, sp_hi)
            sb = rnd.randint(sb_lo, sb_hi)
            chains.append((hub_id, sp, sb))
    # solo / low-degree chains: hub pool indices n_cluster_hubs .. n_cluster_hubs+solo_hub_pool-1
    for _ in range(n_solo):
        h = n_cluster_hubs + rnd.randrange(solo_hub_pool)
        sp = rnd.randint(*SOLO_SP)
        sb = rnd.randint(*SOLO_SB)
        chains.append((h, sp, sb))
    Hn = n_cluster_hubs + solo_hub_pool
    return chains, Hn


def plan(testId):
    """Return (cluster_sizes, cluster_bands, n_solo, solo_hub_pool) per testId."""
    if testId == 1:
        return [5], [BANDS[0]], 3, 2
    if testId == 2:
        return [9], [BANDS[1]], 5, 3
    if testId == 3:
        return [10, 8], [BANDS[0], BANDS[1]], 6, 3
    if testId == 4:
        return [18], [BANDS[0]], 8, 4
    if testId == 5:
        return [18, 16], [BANDS[0], BANDS[1]], 10, 4
    if testId == 6:
        return [20, 18, 15], [BANDS[0], BANDS[1], BANDS[0]], 12, 6
    if testId == 7:
        return [45], [BANDS[1]], 15, 6
    if testId == 8:
        return [30, 30], [BANDS[0], BANDS[1]], 30, 8
    if testId == 9:
        return [35, 35, 30], [BANDS[0], BANDS[1], BANDS[0]], 25, 8
    if testId == 10:
        return [30, 30, 30, 30], [BANDS[0], BANDS[1], BANDS[0], BANDS[1]], 30, 10
    # fallback for any extra testIds
    return [12, 12], [BANDS[0], BANDS[1]], 8, 4


def main():
    testId = int(sys.argv[1])
    rnd = random.Random(1_000_003 * testId + 17)
    cluster_sizes, cluster_bands, n_solo, solo_hub_pool = plan(testId)
    chains, Hn = build(rnd, cluster_sizes, cluster_bands, n_solo, solo_hub_pool)
    C = len(chains)
    m = 2 * C + Hn
    out = [f"{m} {C}"]
    for i, (h, sp, sb) in enumerate(chains):
        p1, p2 = 2 * i, 2 * i + 1
        hub = 2 * C + h
        out.append(f"{p1} {p2} {hub} {sp} {sb}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
