import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(1, 9)
    q = random.randint(1, 14)
    # large magnitudes sometimes, to exercise big sums (overflow-prone),
    # while small n keeps the brute force fast and the cross-check meaningful.
    def rndval():
        if random.random() < 0.5:
            return random.randint(-10**9, 10**9)
        return random.randint(-5, 5)
    a = [rndval() for _ in range(n)]

    lines = []
    lines.append(f"{n} {q}")
    lines.append(" ".join(map(str, a)))
    for _ in range(q):
        t = random.randint(1, 2)
        l = random.randint(1, n)
        r = random.randint(l, n)
        if t == 1:
            v = rndval()
            lines.append(f"1 {l} {r} {v}")
        else:
            lines.append(f"2 {l} {r}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
