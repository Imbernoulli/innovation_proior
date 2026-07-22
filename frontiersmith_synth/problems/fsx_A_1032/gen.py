#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE instance of the banquet-replica-router problem to stdout.
Deterministic: all randomness is seeded ONLY from testId.

Instance =
  K R D B
  cap[0..K-1]
  B lines, each: s d_1 .. d_s   (a "course": the set of dishes plated together)

Design: R fixed replicas required per dish; K kitchens partition (roughly) into
G = K // R disjoint "replica blocks" of R kitchens each. A perfectly
frequency-balanced (consistent-hashing-style) placement assigns dish d to block
(d mod G) -- every kitchen ends up hosting almost exactly the same number of
replicas. The trap: a handful of SIGNATURE tasting menus are repeated many times
across the banquet season, and each signature menu's dishes are drawn from a
SINGLE residue class mod G on purpose -- so a frequency-balanced placement gives
those co-plated dishes IDENTICAL replica sets, and no routing can rescue that.
An antiaffinity-aware placement that reads the course trace and spreads
co-occurring dishes across distinct kitchens avoids the collision entirely.
"""
import sys
import random
import math

R_FIXED = 3

# (K, D, B, courseMin, courseMax, trapFrac, numSignatureMenus)
#   K stretches out (more kitchens => more antiaffinity headroom) while course
#   sizes grow only mildly -- past the point where K exceeds a course's size,
#   a strong antiaffinity placement's per-course floor stops shrinking (it is
#   already down near 1 kitchen-load), so the raw naive-vs-strong gap tracks
#   course size, not K, and never blows past the checker's scoring cap.
PARAMS = {
    1:  (9,   90,   40,  6,  10, 0.14, 2),
    2:  (12,  140,  60,  7,  12, 0.16, 2),
    3:  (15,  200,  80,  8,  14, 0.18, 3),
    4:  (18,  280,  110, 9,  16, 0.19, 3),
    5:  (21,  370,  140, 10, 18, 0.20, 3),
    6:  (24,  480,  180, 10, 18, 0.21, 4),
    7:  (27,  600,  220, 11, 20, 0.22, 4),
    8:  (30,  740,  260, 11, 20, 0.23, 4),
    9:  (33,  900,  300, 12, 22, 0.24, 5),
    10: (36,  1080, 350, 12, 22, 0.25, 5),
}


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    K, D, B, cmin, cmax, trapFrac, numSig = PARAMS[t]
    R = R_FIXED
    G = K // R
    assert G >= 1 and K % R == 0

    rnd = random.Random(1_000_003 * t + 7)

    # residue blocks: blocks[r] = dishes d with d % G == r
    blocks = [[] for _ in range(G)]
    for d in range(D):
        blocks[d % G].append(d)

    # capacity: enough headroom above the perfectly-balanced load to let an
    # antiaffinity placement actually spread hot dishes out, but not so much
    # slack that capacity stops mattering.
    max_block = max(len(b) for b in blocks)
    per_kitchen_balanced = math.ceil(max_block)  # balanced placement load/kitchen
    slack = max(2, per_kitchen_balanced // 14)
    cap_val = per_kitchen_balanced + slack
    cap = [cap_val] * K

    # signature menus: a handful of residues get a FIXED core dish list that is
    # resampled (sub-selected) across many repeated courses -- like a caterer
    # repeating a tasting flight across many banquet nights.
    sig_residues = rnd.sample(range(G), min(numSig, G))
    sig_core = {}
    for r in sig_residues:
        pool = blocks[r]
        core_size = min(len(pool), cmax)
        sig_core[r] = rnd.sample(pool, core_size)

    lines = []
    lines.append(f"{K} {R} {D} {B}")
    lines.append(" ".join(str(c) for c in cap))

    for _ in range(B):
        s = rnd.randint(cmin, cmax)
        if sig_residues and rnd.random() < trapFrac:
            r = rnd.choice(sig_residues)
            core = sig_core[r]
            s = min(s, len(core))
            dishes = rnd.sample(core, s)
        else:
            s = min(s, D)
            dishes = rnd.sample(range(D), s)
        rnd.shuffle(dishes)
        lines.append(f"{len(dishes)} " + " ".join(str(x) for x in dishes))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
