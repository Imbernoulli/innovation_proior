#!/usr/bin/env python3
"""gen.py <testId> -- generator for fsx_B_1087 (gray-code test ordering).

Prints ONE instance to stdout. Deterministic: seeded by testId only.

Hidden structure: configurations live on a hypercube of M binary control lines.
Some lines are EXPENSIVE (one 'master' line plus a few 'regional' lines); the rest
are cheap. Trap cases plant sparse cheap-subclusters inside each expensive block,
plus cross-boundary 'decoy' configs that sit only 2-3 unweighted Hamming flips away
from a subcluster -- so an unweighted nearest-neighbor greedy keeps re-crossing
expensive boundaries (the master line many times), while the cost-aware structure
nests cheap flips inside and toggles the expensive lines nearly monotonically.
Input order is grouped by the master line on structured cases, so the checker's
input-order baseline is not inflated by master flips.
"""
import sys
import random

# tid: (N, use_master, R regional, K cheap, kind, n_decoys, group_input)
CASES = {
    1:  (40,   0, 2, 5,  "uniform", 0,  False),
    2:  (90,   0, 3, 5,  "uniform", 0,  False),
    3:  (170,  0, 3, 6,  "dense",   0,  False),
    4:  (320,  1, 2, 7,  "dense",   0,  True),
    5:  (520,  1, 3, 7,  "trap",    18, True),
    6:  (760,  1, 3, 8,  "trap",    30, True),
    7:  (1000, 1, 3, 8,  "trap",    44, True),
    8:  (1300, 1, 3, 9,  "trap",    58, True),
    9:  (1700, 1, 4, 9,  "trap",    74, True),
    10: (2200, 1, 4, 9,  "trap",    92, True),
}


def hamming(a, b):
    return bin(a ^ b).count("1")


def make_costs(rng, use_master, R, K, kind):
    costs = []
    if kind == "uniform":
        return [rng.randint(1, 8) for _ in range(use_master + R + K)]
    if use_master:
        costs.append(rng.randint(45, 65))           # master line: very expensive
    for _ in range(R):
        costs.append(rng.randint(8, 13))            # regional lines: expensive
    for _ in range(K):
        costs.append(rng.randint(2, 5))             # cheap lines (min 2: no free moves)
    return costs


def sample_centers(rng, K, n_centers, min_dist):
    centers = []
    tries = 0
    while len(centers) < n_centers and tries < 4000:
        tries += 1
        c = rng.getrandbits(K)
        if all(hamming(c, o) >= min_dist for o in centers):
            centers.append(c)
    return centers


def fill_cluster(rng, K, center, used, block, quota):
    """Yield up to `quota` distinct configs in `block` within cheap-Hamming <=2 of center."""
    out = []
    # always include the center itself
    cand = (block << K) | center
    if cand not in used and quota > 0:
        used.add(cand)
        out.append(cand)
        quota -= 1
    tries = 0
    while quota > 0 and tries < 500:
        tries += 1
        d = 2 if rng.random() < 0.65 else 3
        bits = rng.sample(range(K), d)
        p = center
        for b in bits:
            p ^= (1 << b)
        full = (block << K) | p
        if full not in used:
            used.add(full)
            out.append(full)
            quota -= 1
    return out


def main():
    tid = int(sys.argv[1])
    N, use_master, R, K, kind, n_decoys, group_input = CASES[tid]
    rng = random.Random((0x5EED << 16) ^ (tid * 2654435761))
    E = use_master + R
    M = E + K
    costs = make_costs(rng, use_master, R, K, kind)

    used = set()
    cfgs = []

    if kind == "uniform":
        while len(cfgs) < N:
            c = rng.getrandbits(M)
            if c not in used:
                used.add(c)
                cfgs.append(c)
    elif kind == "dense":
        nblocks = 1 << E
        per = [N // nblocks] * nblocks
        for i in range(N - (N // nblocks) * nblocks):
            per[i] += 1
        for blk in range(nblocks):
            got = 0
            while got < per[blk]:
                p = rng.getrandbits(K)
                full = (blk << K) | p
                if full not in used:
                    used.add(full)
                    cfgs.append(full)
                    got += 1
    else:  # trap
        nblocks = 1 << E
        target = N - n_decoys
        # 2 subcluster centers per block, min cheap-Hamming 5 apart
        centers = {}
        slots = []  # (block, center) round-robin slots
        for blk in range(nblocks):
            cs = sample_centers(rng, K, 2, 5)
            centers[blk] = cs
            for c in cs:
                slots.append((blk, c))
        # round-robin fill until `target` configs
        idx = 0
        while len(cfgs) < target:
            blk, c = slots[idx % len(slots)]
            idx += 1
            got = fill_cluster(rng, K, c, used, blk, 1)
            cfgs.extend(got)
            if idx > 40 * len(slots) * 60:  # safety, should not trigger
                break
        # plant decoys: configs just across an expensive boundary (master with
        # prob 0.6, else a regional line), cheap-close to some block's subcluster
        planted = 0
        tries = 0
        while planted < n_decoys and tries < 200 * n_decoys + 200:
            tries += 1
            blk = rng.randrange(nblocks)
            c = centers[blk][rng.randrange(len(centers[blk]))]
            if use_master and rng.random() < 0.6:
                b = 0
            else:
                b = rng.randrange(use_master, E) if use_master else rng.randrange(E)
            nblk = blk ^ (1 << b)
            p = c ^ (1 << rng.randrange(K))
            full = (nblk << K) | p
            if full not in used:
                used.add(full)
                cfgs.append(full)
                planted += 1
        N = len(cfgs)

    # input order
    order = list(range(len(cfgs)))
    if group_input and use_master:
        g0 = [i for i in order if not ((cfgs[i] >> 0) & 1)]
        g1 = [i for i in order if (cfgs[i] >> 0) & 1]
        rng.shuffle(g0)
        rng.shuffle(g1)
        order = g0 + g1
    else:
        rng.shuffle(order)

    out = ["%d %d" % (len(cfgs), M), " ".join(map(str, costs))]
    for i in order:
        c = cfgs[i]
        out.append("".join('1' if (c >> j) & 1 else '0' for j in range(M)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
