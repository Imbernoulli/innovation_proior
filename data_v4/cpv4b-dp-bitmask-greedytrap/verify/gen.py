import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep brute force (2^k subsets) tractable: k small.
    m = rng.randint(0, 5)
    k = rng.randint(0, 12)

    lines = []
    lines.append(f"{m} {k}")
    for _ in range(k):
        c = rng.randint(1, 50)
        # random non-empty-ish subset of channels (may be empty)
        if m == 0:
            t = 0
            chans = []
        else:
            t = rng.randint(0, m)
            chans = rng.sample(range(m), t)
        parts = [str(c), str(t)] + [str(x) for x in chans]
        lines.append(" ".join(parts))
    sys.stdout.write("\n".join(lines) + "\n")

main()
