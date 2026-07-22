#!/usr/bin/env python3
"""gen.py <testId> -- prints one 'moonshine still' cut-cascade instance to stdout.
Deterministic: all randomness seeded from testId only.

Each component c is laid out as n_blobs SEPARATE lobes ("re-condensation pockets") of
width `blob_width`, interleaved round-robin with the other components' lobes along the
G-bin axis (so any single lobe is a small, locally-pure, but geographically ISOLATED
pocket of its component). Adjacent lobes (of different components) blend slightly at
their shared edge, so the feedstock genuinely overlaps there (no lobe is perfectly
pure) -- but each lobe is still individually recognizable as "mostly component c".

Pricing has a batch-size qualifier on top of the purity band: a fraction whose mass is
below M_min is capped at a small multiplier REGARDLESS of purity (a boutique/small-lot
penalty -- selling a thimble of 98%-pure spirit still only fetches a hobbyist price;
only a properly-sized batch clears certification for the premium band). A single lobe's
mass is deliberately sized just under M_min, so cutting it out and selling it alone
never earns full credit for its purity -- but consolidating several SAME-component
lobes (scattered, non-adjacent) into one recycle stream crosses M_min and unlocks the
full purity-band price for the combined batch.
"""
import sys, random


def build(testId):
    # ---- difficulty ladder: (K, n_blobs, blob_width) ----
    ladder = [
        (3, 2, 3),
        (3, 2, 4),
        (3, 3, 3),
        (4, 2, 3),
        (4, 2, 4),
        (4, 3, 3),
        (5, 2, 4),
        (5, 3, 3),
        (5, 4, 2),
        (5, 5, 2),
    ]
    idx = testId - 1
    idx = min(max(idx, 0), len(ladder) - 1)
    K, n_blobs, blob_width = ladder[idx]
    G = K * n_blobs * blob_width
    # pass-1 cuts may land on any bin boundary (kept as a general mechanism but not the
    # primary trap lever here); step1=1 means no coarse restriction is in force.
    step1 = 1

    rng = random.Random(1000003 * testId + 17)

    peak = [80.0 * (0.7 + 0.6 * rng.random()) for _ in range(K)]
    # value per unit mass at 100% purity
    v = [round(3.0 + 6.0 * rng.random(), 3) for _ in range(K)]

    n_slots = n_blobs * K
    m = [[0.0] * G for _ in range(K)]
    for s in range(n_slots):
        c = s % K
        lo = s * blob_width
        hi = lo + blob_width
        for g in range(lo, hi):
            val = peak[c]
            # blend a fraction of each edge bin with the neighboring slot's component
            if blob_width >= 2 and g == lo:
                val *= 0.92
            if blob_width >= 2 and g == hi - 1:
                val *= 0.92
            jitter = 1.0 + 0.03 * (rng.random() * 2 - 1)
            m[c][g] = round(max(0.0, val * jitter), 4)
        # deposit the blended-away mass into the neighboring slot's component
        if blob_width >= 2:
            prev_c = (s - 1) % K
            next_c = (s + 1) % K
            edge_mass = 0.08 * peak[c]
            if lo - 1 >= 0:
                jitter = 1.0 + 0.03 * (rng.random() * 2 - 1)
                m[prev_c][lo] = round(m[prev_c][lo] + max(0.0, edge_mass * jitter), 4)
            if hi < G:
                jitter = 1.0 + 0.03 * (rng.random() * 2 - 1)
                m[next_c][hi - 1] = round(m[next_c][hi - 1] + max(0.0, edge_mass * jitter), 4)

    # economics
    H = round(0.35 + 0.1 * rng.random(), 3)
    energyCost = round((0.03 + 0.03 * rng.random()) * (sum(v) / K), 3)

    # batch-size price qualifier: a single lobe's own mass is ~peak_c*blob_width;
    # M_min sits comfortably above one lobe but well below n_blobs>=2 lobes combined.
    avg_peak = sum(peak) / K
    M_min = round(1.55 * avg_peak * blob_width, 2)
    cap_small = 0.20

    # Note the wide low-purity band: two adjacent (necessarily different-component,
    # round-robin) lobes merged together land near 1/2 (or lower, for 3+) purity --
    # comfortably UNDER 0.60 -- so that merge is worse than the small-lot cap. There
    # is no way to dodge the batch-size penalty except by consolidating same-dominant
    # lobes, which stay high-purity and land in a much better band.
    bands = [(0.0, 0.15), (0.6, 0.20), (0.75, 0.55), (0.88, 1.00)]

    out = []
    out.append(f"{K} {G} {step1}")
    out.append(" ".join(f"{x:.3f}" for x in v))
    for c in range(K):
        out.append(" ".join(f"{x:.4f}" for x in m[c]))
    out.append(f"{H:.3f} {energyCost:.3f}")
    out.append(f"{M_min:.3f} {cap_small:.3f}")
    out.append(str(len(bands)))
    for lo, mult in bands:
        out.append(f"{lo:.3f} {mult:.3f}")
    return "\n".join(out) + "\n"


def main():
    testId = int(sys.argv[1])
    sys.stdout.write(build(testId))


if __name__ == "__main__":
    main()
