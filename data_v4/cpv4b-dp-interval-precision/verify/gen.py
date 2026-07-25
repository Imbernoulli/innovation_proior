import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 8)
    K = rng.randint(1, n)

    # keep magnitudes small for brute, but occasionally push the ratio to extremes
    bmax = rng.choice([1, 2, 3, 5, 10])
    wmax = rng.choice([1, 2, 3, 5, 10])

    out = []
    out.append(f"{n} {K}")
    for _ in range(n):
        b = rng.randint(-bmax, bmax)
        w = rng.randint(1, wmax)
        out.append(f"{b} {w}")

    # ratio p/q for admissibility |B|*q >= p*W
    # choose p,q so that some groups pass and some fail
    p = rng.randint(0, 6)
    q = rng.randint(1, 6)
    out.append(f"{p} {q}")

    sys.stdout.write("\n".join(out) + "\n")

main()
