import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Bias toward small / corner cases that stress sign & base-case handling.
    n = rng.randint(1, 7)

    # Values: deliberately include many negatives and zeros, occasional positives.
    # Sometimes force an ALL-negative instance to exercise the "answer may be negative" corner.
    mode = rng.randint(0, 3)
    v = []
    for i in range(n):
        if mode == 0:
            # all negative
            v.append(rng.randint(-9, -1))
        elif mode == 1:
            # negatives and zeros only
            v.append(rng.randint(-9, 0))
        elif mode == 2:
            # mixed including positives
            v.append(rng.randint(-9, 9))
        else:
            # mostly zeros
            v.append(rng.choice([0, 0, 0, rng.randint(-9, 9)]))

    # Edges go strictly deeper: a < b. Pick a random subset of forward edges.
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.45:
                edges.append((a, b))
    rng.shuffle(edges)
    m = len(edges)

    out = []
    out.append(f"{n} {m}")
    out.append(" ".join(str(x) for x in v))
    for (a, b) in edges:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

main()
