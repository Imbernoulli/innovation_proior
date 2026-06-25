import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to stress the precision/overflow path:
    #  - small n so the O(n^2) brute is fast;
    #  - a "tight" mode that draws values and p/q from a small set so many ratios sit exactly on the
    #    threshold (the adversarial near-equal case where double rounding flips the answer);
    #  - a "wide" mode with large values so cross-products approach 10^18.
    mode = rng.randint(0, 2)
    n = rng.randint(0, 12)

    if mode == 0:
        # tight: small alphabet, threshold also small -> exact-equality collisions are common
        vmax = rng.choice([1, 2, 3, 4, 6])
        vals = [rng.randint(1, vmax) for _ in range(n)]
        q = rng.randint(1, 4)
        p = rng.randint(q, 4 * q)
    elif mode == 1:
        # wide: values and tolerance up to 4*10^9 -> cross-products near 1.6*10^19, int64 overflow
        VMAX = 4 * 10**9
        vals = [rng.randint(1, VMAX) for _ in range(n)]
        q = rng.randint(1, VMAX)
        p = rng.randint(q, VMAX)
    else:
        # mixed: medium values, threshold chosen to land near a real ratio in the data
        vals = [rng.randint(1, 1000) for _ in range(n)]
        q = rng.randint(1, 1000)
        p = rng.randint(q, 1000)

    out = []
    out.append(f"{n} {p} {q}")
    out.append(" ".join(map(str, vals)))
    sys.stdout.write("\n".join(out) + "\n")

main()
