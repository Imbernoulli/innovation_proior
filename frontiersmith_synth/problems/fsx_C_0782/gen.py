import sys, random

# ---- palate-adaptation flight-order instance generator --------------------
# N wine samples have true intensities V_1..V_N (1..100). The judge tastes
# them in a chosen order (with up to K "palate-cleanser" resets spliced in).
# Two mechanisms distort what the judge perceives at each tasting:
#   (1) gain-control-adaptation: a running EMA `a` of recently-tasted true
#       intensities compresses perceptual gain whenever `a` drifts far from
#       the neutral center -- deep drift into one register saturates the
#       palate and every subsequent reading is squashed toward `a`.
#   (2) sequential-contrast-carryover: the reading is further displaced in
#       the direction of the jump from the immediately preceding tasted
#       sample -- a big jump exaggerates the current reading away from truth.
# A handful of samples are "flagship" (high judging weight); their perceived
# error counts for much more of the score.
#
# TRAP testIds (6,7,8) plant a long contiguous run of same-extreme values
# (a block near VMAX and a block near VMIN) with several flagships seeded
# INSIDE the extreme block. The "obvious" recipe -- sort ascending, the
# monotone sommelier sweep, spending cleansers evenly -- minimizes
# contrast jumps but marches the adaptation EMA straight into saturation
# across the whole extreme block, crushing the flagship readings sitting in
# it. The band-limited zigzag (pair the i-th smallest with the i-th
# largest-of-upper-half) keeps `a` oscillating near the center instead, and
# a value-aware cleanser budget resets gain exactly before the flagships.


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20260714 + 104729 * testId)

    sizes = [8, 12, 20, 30, 50, 80, 120, 160, 220, 300]
    N = sizes[testId - 1]
    K = max(1, N // 10)

    ALPHA = 300              # EMA rate, scaled by 1000 (0.300)
    D = 20000                # gain-saturation elbow, scaled by 1000 (20.0 true
                              # units); the checker raises D and the drift `d`
                              # to the 3rd power, so gain stays open for
                              # ordinary drift and only truly collapses once a
                              # run pushes the adaptation level far past D.
    CC_NUM, CC_DEN = 1, 8    # contrast coefficient 1/8
    CENTER = 50

    is_trap = testId in (6, 7, 8)

    V = [0] * (N + 1)
    if is_trap:
        n_high = N // 2
        n_low = N // 4
        n_mid = N - n_high - n_low
        vals = []
        vals += [rng.randint(88, 100) for _ in range(n_high)]
        vals += [rng.randint(1, 12) for _ in range(n_low)]
        vals += [rng.randint(42, 58) for _ in range(n_mid)]
        rng.shuffle(vals)
        for i in range(1, N + 1):
            V[i] = vals[i - 1]
    else:
        # ordinary flights stay in a moderate band -- no planted saturation
        # run, so the sommelier sweep's small consecutive jumps are a
        # genuinely good, low-risk choice here.
        for i in range(1, N + 1):
            V[i] = rng.randint(30, 70)

    W = [1] * (N + 1)
    n_flag = max(1, N // 12)
    if is_trap:
        pool = [i for i in range(1, N + 1) if V[i] >= 88 or V[i] <= 12]
        if not pool:
            pool = list(range(1, N + 1))
        chosen = rng.sample(pool, min(n_flag, len(pool)))
    else:
        chosen = rng.sample(range(1, N + 1), min(n_flag, N))
    for idx in chosen:
        W[idx] = rng.randint(15, 20)

    print(N, K)
    print(ALPHA, D, CC_NUM, CC_DEN, CENTER)
    print(*[V[i] for i in range(1, N + 1)])
    print(*[W[i] for i in range(1, N + 1)])


if __name__ == "__main__":
    main()
