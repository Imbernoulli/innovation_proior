import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to stress sign handling and corners.
    regime = rng.randint(0, 6)

    if regime == 0:
        # tiny / degenerate dimensions including 0
        R = rng.randint(0, 2)
        C = rng.randint(0, 2)
    elif regime == 1:
        R = 1
        C = rng.randint(1, 6)
    elif regime == 2:
        C = 1
        R = rng.randint(1, 6)
    else:
        R = rng.randint(1, 5)
        C = rng.randint(1, 5)

    N = R * C

    # Value distribution biased toward 0 and small magnitudes so that
    # sources (<0), conductors (==0), and walls (>0) mix densely.
    vals = []
    for _ in range(N):
        if regime == 4:
            # no strictly-negative cell: only zeros and positives (answer -1)
            v = rng.choice([0, rng.randint(1, 5)])
        elif regime == 5:
            # all non-positive (sources + conductors, no walls)
            v = rng.randint(-5, 0)
        elif regime == 6:
            # zeros-heavy: long conductor channels between sparse sources
            r = rng.random()
            if r < 0.55:
                v = 0
            elif r < 0.78:
                v = -rng.randint(1, 9)
            else:
                v = rng.randint(1, 9)
        else:
            r = rng.random()
            if r < 0.28:
                v = 0
            elif r < 0.60:
                v = -rng.randint(1, 9)
            else:
                v = rng.randint(1, 9)
        vals.append(v)

    out = [f"{R} {C}"]
    if N > 0:
        out.append(" ".join(str(v) for v in vals))
    sys.stdout.write("\n".join(out) + "\n")

main()
