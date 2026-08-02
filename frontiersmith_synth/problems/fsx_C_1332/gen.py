#!/usr/bin/env python3
"""gen.py <testId> -- print ONE fragrance-blend-trajectory instance to stdout.

Instance: K candidate ingredients, each with a 4-axis odor descriptor vector
(citrus/floral/woody/musk intensity-per-unit-concentration), a volatility
class expressed as an exponential evaporation rate k_i (1/hour), and a
per-ingredient IFRA regulatory concentration cap cap_i. A KxK perceptual
masking table mask[i][j] says how much RAW intensity of ingredient j
suppresses the PERCEIVED intensity of ingredient i when both are present.
A target scent profile is given at 5 checkpoints across four hours
(t=0,1,2,3,4h), shaped as a top-note -> heart-note -> base-note pyramid
(three different dominant axes at t=0, t=2, t=4).

Every instance deliberately plants four structurally distinct ingredients:
  0: TOP_STAR    -- near-perfect match to the t=0 target axis, fast decay
                    (k in [2.4,4.0]/hr -> ~gone by hour 2), generous cap.
  1: BASE_STAR   -- near-perfect match to the t=4 target axis, slow decay
                    (k in [0.05,0.22]/hr -> barely moves in 4h), LOW overlap
                    with the t=0 axis (so a t=0-only fit under-weights it),
                    and gets heavily masked BY TOP_STAR while it's still
                    around (mask[BASE_STAR][TOP_STAR] is large).
  2: HEART_STAR  -- matches the t=2 target axis, medium decay.
  3: VERSATILE   -- decent overlap with BOTH the t=0 and t=4 axes, but a
                    very tight IFRA cap, so it cannot single-handedly cover
                    the trajectory.
Remaining ingredients (if K>4) are unstructured filler.

A solver that only matches the t=0 snapshot (ignoring decay and masking)
will load up on TOP_STAR and roughly ignore BASE_STAR/HEART_STAR -- it
scores well at t=0 and collapses by t=2..4h, which is exactly the trap
this family is meant to expose. Later testIds widen K (more filler /
search noise), tighten VERSATILE's cap, and strengthen the background
masking, so the trap gets sharper, not weaker.

Everything is seeded by testId only -> bit-for-bit reproducible.
"""
import sys
import random

D = 4  # AXES = citrus, floral, woody, musk
T = 5
TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]  # hours, spans the full 4h window

K_SCHEDULE = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8]


def mix(a, b, s):
    return [a[i] * (1.0 - s) + b[i] * s for i in range(D)]


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260802 + 104729 * tid)
    K = K_SCHEDULE[(tid - 1) % len(K_SCHEDULE)]
    difficulty = (tid - 1) / 9.0  # 0..1, ramps trap sharpness

    axis_perm = list(range(D))
    rng.shuffle(axis_perm)
    axis_top, axis_heart, axis_base = axis_perm[0], axis_perm[1], axis_perm[2]

    # Per-ingredient IFRA caps are scaled up by CAP_SCALE relative to a
    # "tight" baseline: this keeps the total available budget (1.0) as the
    # real scarce resource that forces genuine multi-ingredient trade-offs,
    # while VERSATILE stays deliberately tight regardless.
    CAP_SCALE = 2.2

    ingredients = []  # list of (desc[D], k, cap)

    # 0: TOP_STAR
    desc = [round(rng.uniform(0.03, 0.14), 4) for _ in range(D)]
    desc[axis_top] = round(rng.uniform(0.80, 0.95), 4)
    k = round(rng.uniform(2.4, 4.0), 4)
    cap = round(rng.uniform(0.24, 0.40) * CAP_SCALE, 4)
    ingredients.append((desc, k, cap))

    # 1: BASE_STAR
    desc = [round(rng.uniform(0.03, 0.14), 4) for _ in range(D)]
    desc[axis_base] = round(rng.uniform(0.80, 0.95), 4)
    desc[axis_top] = round(rng.uniform(0.02, 0.08), 4)
    k = round(rng.uniform(0.05, 0.22), 4)
    cap = round(rng.uniform(0.20, 0.38) * CAP_SCALE, 4)
    ingredients.append((desc, k, cap))

    # 2: HEART_STAR
    desc = [round(rng.uniform(0.03, 0.14), 4) for _ in range(D)]
    desc[axis_heart] = round(rng.uniform(0.75, 0.92), 4)
    k = round(rng.uniform(0.6, 1.3), 4)
    cap = round(rng.uniform(0.20, 0.38) * CAP_SCALE, 4)
    ingredients.append((desc, k, cap))

    # 3: VERSATILE -- overlaps both top and base axes, tight (and tightening) cap
    desc = [round(rng.uniform(0.02, 0.10), 4) for _ in range(D)]
    desc[axis_top] = round(rng.uniform(0.38, 0.52), 4)
    desc[axis_base] = round(rng.uniform(0.38, 0.52), 4)
    k = round(rng.uniform(0.30, 0.60), 4)
    cap = round(max(0.018, rng.uniform(0.080, 0.100) - 0.0032 * tid), 4)
    ingredients.append((desc, k, cap))

    # fillers
    for _ in range(K - 4):
        desc = [round(rng.uniform(0.05, 0.45), 4) for _ in range(D)]
        k = round(rng.uniform(0.10, 3.5), 4)
        cap = round(rng.uniform(0.06, 0.30) * CAP_SCALE, 4)
        ingredients.append((desc, k, cap))

    # masking table: background noise + one planted strong directional term
    base_bg = 0.10 + 0.018 * tid
    mask = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            mask[i][j] = round(rng.uniform(0.0, base_bg), 4)
    # BASE_STAR (1) is strongly masked while TOP_STAR (0) is still loud.
    mask[1][0] = round(rng.uniform(0.75, 1.25) + 0.35 * difficulty, 4)

    # target trajectory: top-note -> heart-note -> base-note pyramid over 4h.
    # BG/PEAK were tuned (via floor-vs-baseline search) so a trajectory- and
    # masking-aware blend can genuinely close most of the gap while a
    # t=0-only fit cannot -- too large a peak/background gap is physically
    # unreachable (raw intensity only decays, never grows), which would
    # flatten every strategy's score together.
    BG, PEAK = 0.14, 0.30
    g0 = [round(rng.uniform(BG * 0.8, BG * 1.2), 4) for _ in range(D)]
    g0[axis_top] = round(rng.uniform(PEAK * 0.85, PEAK * 1.05), 4)
    gmid = [round(rng.uniform(BG * 0.8, BG * 1.2), 4) for _ in range(D)]
    gmid[axis_heart] = round(rng.uniform(PEAK * 0.85, PEAK * 1.05), 4)
    gend = [round(rng.uniform(BG * 0.8, BG * 1.2), 4) for _ in range(D)]
    gend[axis_base] = round(rng.uniform(PEAK * 0.85, PEAK * 1.05), 4)

    targets = [g0, mix(g0, gmid, 0.5), gmid, mix(gmid, gend, 0.5), gend]

    out = []
    out.append("%d %d %d" % (K, D, T))
    for desc, k, cap in ingredients:
        out.append(" ".join("%.4f" % v for v in desc) + " %.5f %.5f" % (k, cap))
    for i in range(K):
        out.append(" ".join("%.4f" % mask[i][j] for j in range(K)))
    out.append(" ".join("%.2f" % t for t in TIMES))
    for g in targets:
        out.append(" ".join("%.4f" % v for v in g))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
