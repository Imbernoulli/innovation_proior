import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 6)
    m = rng.randint(0, 12)
    lines = ["%d %d" % (n, m)]
    for _ in range(m):
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        lines.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(lines) + "\n")

main()
