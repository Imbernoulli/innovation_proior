import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to stress negatives/zeros, all-negative, empty, tight bands.
    regime = rng.randint(0, 6)

    if regime == 0:
        n = 0
    elif regime == 1:
        n = 1
    elif regime == 2:
        n = rng.randint(2, 8)
    else:
        n = rng.randint(0, 12)

    # value range, sometimes all-negative or all-zero or large-magnitude
    vr = rng.randint(0, 4)
    vals = []
    for _ in range(n):
        if vr == 0:
            vals.append(rng.randint(-6, 6))
        elif vr == 1:
            vals.append(rng.randint(-8, -1))   # all negative
        elif vr == 2:
            vals.append(0)                     # all zero
        elif vr == 3:
            vals.append(rng.randint(-3, 3))
        else:
            vals.append(rng.randint(-10**9, 10**9))  # large magnitude

    # band [lo, hi]; sometimes lo > hi (empty band), sometimes single value
    if vr == 4:
        lo = rng.randint(-2 * 10**9, 2 * 10**9)
        span = rng.randint(-3, 4 * 10**9)
    else:
        lo = rng.randint(-16, 16)
        span = rng.randint(-2, 8)   # negative span -> lo > hi (empty band)
    hi = lo + span

    out = [f"{n} {lo} {hi}"]
    out.append(" ".join(str(v) for v in vals))
    sys.stdout.write("\n".join(out) + "\n")

main()
