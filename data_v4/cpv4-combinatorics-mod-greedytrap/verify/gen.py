import sys
import random

# Random SMALL-case generator parameterized by an integer seed.
# Keeps a+b tiny so the exponential brute force stays fast, and exercises the
# corners: a<b (sometimes infeasible), m=0 (tight), large m (no constraint).

# The problem guarantees p is a prime strictly greater than the maximum a+b
# (here a+b <= 16 in tiny cases), so Fermat-based factorial inverses are valid.
# Including a few "just barely large enough" primes stresses that boundary.
PRIMES = [17, 19, 23, 29, 31, 1000003, 1000000007, 998244353]

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    q = rng.randint(1, 6)
    p = rng.choice(PRIMES)
    lines = [f"{q} {p}"]
    for _ in range(q):
        a = rng.randint(0, 8)
        b = rng.randint(0, 8)
        # bias m to small values to stress the binding constraint, occasionally large
        if rng.random() < 0.75:
            m = rng.randint(0, 4)
        else:
            m = rng.randint(0, 12)
        lines.append(f"{a} {b} {m}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
