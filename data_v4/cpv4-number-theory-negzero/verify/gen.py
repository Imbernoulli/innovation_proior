import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # tiny cases to stress the corners: negatives, zeros, all-zero subranges
    n = rng.randint(1, 8)
    q = rng.randint(1, 8)

    # value pool emphasises zeros and small magnitudes so gcd structure varies,
    # and includes negatives so sign handling is exercised.
    pool = [-12, -6, -4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4, 6, 12]
    a = [rng.choice(pool) for _ in range(n)]

    lines = []
    lines.append(f"{n} {q}")
    lines.append(" ".join(str(x) for x in a))
    for _ in range(q):
        l = rng.randint(1, n)
        r = rng.randint(1, n)
        if l > r:
            l, r = r, l
        lines.append(f"{l} {r}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
