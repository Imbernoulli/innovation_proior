import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 7)            # small enough for permutation brute force

    # Pick a regime per seed to exercise different precision corners.
    regime = rng.randint(0, 3)
    lines = [str(n)]
    for _ in range(n):
        if regime == 0:
            # tiny values: plain correctness
            p = rng.randint(1, 6)
            w = rng.randint(1, 6)
        elif regime == 1:
            # large values near 10^9: cross-product near 10^18, objective huge
            p = rng.randint(1, 10**9)
            w = rng.randint(1, 10**9)
        elif regime == 2:
            # near-tied ratios: p/w almost equal -> doubles would misorder.
            # base ratio a/b, jitter the numerator/denominator slightly.
            a = rng.randint(1, 31623)    # ~sqrt(1e9)
            b = rng.randint(1, 31623)
            k = rng.randint(10000, 31000)
            p = a * k + rng.randint(-1, 1)
            w = b * k + rng.randint(-1, 1)
            p = max(1, min(p, 10**9))
            w = max(1, min(w, 10**9))
        else:
            # equal ratios (exact ties) mixed with large values
            base_p = rng.randint(1, 30000)
            base_w = rng.randint(1, 30000)
            m = rng.randint(1, 33000)
            p = base_p * m
            w = base_w * m
            p = max(1, min(p, 10**9))
            w = max(1, min(w, 10**9))
        lines.append(f"{p} {w}")

    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
