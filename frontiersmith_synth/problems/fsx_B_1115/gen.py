import sys, random


def make_case(i):
    rng = random.Random(19151 + 97 * i)

    # difficulty ladder: T grows 4 -> 12 across the 10 cases
    T = min(4 + (i - 1), 12)

    # regime selection:
    #  case 1        : easy sanity (mild decay & scrap everywhere)
    #  cases 2..8    : the compounding-decay trap -- S1,S2,S3 all sizeable AND
    #                  d1,d2,d3 all drawn from the SAME moderate-high range, so
    #                  batching *any single* stage looks attractive evaluated in
    #                  isolation, but batching several stages at once compounds
    #                  queue time (and hence decay loss) multiplicatively through
    #                  the chain -- the innovation hook.
    #  case 9        : skewed/staircase decay (buffers have very different rates)
    #                  -- rewards a genuinely different lot-size RATIO between
    #                  each pair of adjacent stages, not a single shared batch size
    #  case 10       : largest / adversarial, clustered-then-sparse demand
    if i == 1:
        regime = "easy"
    elif i <= 8:
        regime = "trap"
    elif i <= 9:
        regime = "staircase"
    else:
        regime = "adversarial"

    if regime == "easy":
        gap_lo, gap_hi = 7, 16
        d1 = rng.uniform(0.002, 0.006)
        d2 = rng.uniform(0.002, 0.006)
        d3 = rng.uniform(0.002, 0.006)
        S1 = rng.randint(50, 100)
        S2 = rng.randint(50, 100)
        S3 = rng.randint(50, 100)
        D_lo, D_hi = 80, 200
    elif regime == "trap":
        gap_lo, gap_hi = 11, 24
        d1 = rng.uniform(0.010, 0.021)
        d2 = rng.uniform(0.010, 0.021)
        d3 = rng.uniform(0.010, 0.021)
        S1 = rng.randint(330, 490)
        S2 = rng.randint(330, 490)
        S3 = rng.randint(330, 490)
        D_lo, D_hi = 100, 230
    elif regime == "staircase":
        gap_lo, gap_hi = 13, 28
        d1 = rng.uniform(0.0006, 0.003)
        d2 = rng.uniform(0.007, 0.018)
        d3 = rng.uniform(0.035, 0.070)
        S1 = rng.randint(260, 460)
        S2 = rng.randint(260, 460)
        S3 = rng.randint(260, 460)
        D_lo, D_hi = 100, 230
    else:  # adversarial: largest T, clustered-then-sparse demand
        gap_lo, gap_hi = None, None
        d1 = rng.uniform(0.011, 0.023)
        d2 = rng.uniform(0.011, 0.023)
        d3 = rng.uniform(0.011, 0.023)
        S1 = rng.randint(300, 470)
        S2 = rng.randint(300, 470)
        S3 = rng.randint(300, 470)
        D_lo, D_hi = 100, 280

    times = [0]
    if regime == "adversarial":
        half = T // 2
        for k in range(1, T):
            gap = rng.randint(4, 10) if k <= half else rng.randint(28, 45)
            times.append(times[-1] + gap)
    else:
        for k in range(1, T):
            times.append(times[-1] + rng.randint(gap_lo, gap_hi))

    demands = [rng.randint(D_lo, D_hi) for _ in range(T)]

    out = [str(T)]
    out.append(" ".join(str(t) for t in times))
    out.append(" ".join(str(d) for d in demands))
    out.append("%d %d %d" % (S1, S2, S3))
    out.append("%.6f %.6f %.6f" % (d1, d2, d3))
    return "\n".join(out) + "\n"


def main():
    i = int(sys.argv[1])
    sys.stdout.write(make_case(i))


if __name__ == "__main__":
    main()
