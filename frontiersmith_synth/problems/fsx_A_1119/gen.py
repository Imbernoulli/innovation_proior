#!/usr/bin/env python3
"""gen.py <testId> -- prints one solidification-front instance to stdout.
Deterministic: seeded ONLY by testId. Builds seeds spread across a line, then
constructs an ACHIEVABLE target microstructure by running a reference cooling
schedule through the shared physics engine (sim.py) that deliberately gives
each grain an asymmetric (skewed) share of its neighboring gap -- so the
target cannot be reached by a "fair" equal-rate multi-front race.
"""
import sys
import random

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import sim

H0 = 100
F = 45
CSTEP = 45


def build(test_id):
    rng = random.Random(20000 + test_id)

    N = 22 + 5 * (test_id - 1)          # 22 .. 67
    P = max(2, round(N / 7.0))
    P = min(P, N // 3)                  # keep gaps meaningfully wide
    P = max(P, 2)

    # seed positions: pinned at both ends, remaining P-2 spread with jitter
    if P == 2:
        positions = [0, N - 1]
    else:
        positions = [0]
        span = N - 1
        for k in range(1, P - 1):
            base = round(k * span / (P - 1))
            jitter = rng.randint(-1, 1)
            positions.append(base + jitter)
        positions.append(N - 1)
        # enforce strictly increasing with a minimum gap of 3
        for k in range(1, len(positions)):
            if positions[k] <= positions[k - 1] + 2:
                positions[k] = positions[k - 1] + 3
        # rescale if we overshot N-1
        if positions[-1] > N - 1:
            overflow = positions[-1] - (N - 1)
            # compress by shifting interior points down proportionally
            for k in range(1, len(positions) - 1):
                shrink = round(overflow * k / (len(positions) - 1))
                positions[k] -= shrink
            positions[-1] = N - 1
            for k in range(1, len(positions)):
                if positions[k] <= positions[k - 1]:
                    positions[k] = positions[k - 1] + 1

    orientations = list(range(1, P + 1))
    seeds = list(zip(positions, orientations))

    # per-gap skew fraction (share claimed by the LEFT seed of the gap),
    # deliberately far from 0.5 so a fair equal-rate race lands wrong.
    skew_pool = [0.15, 0.18, 0.22, 0.25, 0.28, 0.72, 0.75, 0.78, 0.82, 0.85]

    heat, solid, orient = sim.new_state(N, seeds, H0)
    total_stages = 0

    def grow_from(seed_pos, count, from_left):
        nonlocal total_stages
        done = 0
        guard = 0
        while done < count:
            if from_left:
                target = sim.frontier_from_left(solid, seed_pos, N)
            else:
                target = sim.frontier_from_right(solid, seed_pos, N)
            if target is None:
                break
            was_solid_before = solid[target]
            while not solid[target]:
                sim.step(heat, solid, orient, target, F, CSTEP)
                total_stages += 1
                guard += 1
                if guard > 40 * N:
                    raise RuntimeError("reference schedule stuck")
            done += 1

    for g in range(P - 1):
        pL, oL = positions[g], orientations[g]
        pR, oR = positions[g + 1], orientations[g + 1]
        gapwidth = pR - pL - 1
        if gapwidth <= 0:
            continue
        f = rng.choice(skew_pool)
        Lcount = round(f * gapwidth)
        Lcount = max(1, min(gapwidth - 1, Lcount)) if gapwidth >= 2 else gapwidth
        Rcount = gapwidth - Lcount
        if Lcount > 0:
            grow_from(pL, Lcount, from_left=True)
        if Rcount > 0:
            grow_from(pR, Rcount, from_left=False)

    assert all(solid), "reference schedule failed to fill the grid"
    T = orient[:]
    for pos, o in seeds:
        assert T[pos] == o

    K = int((total_stages * 1.35) // 1) + 12

    return N, K, seeds, T


def main():
    test_id = int(sys.argv[1])
    N, K, seeds, T = build(test_id)
    out = []
    out.append(f"{N} {K} {H0} {F} {CSTEP}")
    out.append(f"{len(seeds)}")
    for pos, o in seeds:
        out.append(f"{pos} {o}")
    out.append(" ".join(str(x) for x in T))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
