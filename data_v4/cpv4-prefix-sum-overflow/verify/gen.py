import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # Tiny cases respecting the contract (1 <= n, 1 <= q).
    n = random.randint(1, 8)
    q = random.randint(1, 8)

    # Mix of magnitudes including large values to exercise (in spirit) the
    # overflow path, plus negatives and zeros.
    pool_choices = [
        lambda: random.randint(-9, 9),
        lambda: random.randint(-1000000000, 1000000000),
        lambda: 0,
    ]
    a = []
    for _ in range(n):
        a.append(random.choice(pool_choices)())

    lines = []
    lines.append(f"{n} {q}")
    lines.append(" ".join(str(x) for x in a))
    for _ in range(q):
        l = random.randint(1, n)
        r = random.randint(l, n)
        lines.append(f"{l} {r}")

    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
