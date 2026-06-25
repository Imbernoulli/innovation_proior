import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases, deliberately biased toward negatives and small m so that
    # negative prefix sums and congruent buckets occur frequently.
    n = rng.randint(0, 12)
    m = rng.randint(1, 6)
    # values can be negative, zero, or positive
    lo, hi = -8, 8
    a = [rng.randint(lo, hi) for _ in range(n)]

    out = [f"{n} {m}"]
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

main()
