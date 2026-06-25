import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # small n so brute (subset * permutation feasibility) is cheap
    n = rng.randint(0, 8)
    # small t and d so that overflow is irrelevant but ties/overflows of
    # the schedule are common (forces many evictions)
    maxt = rng.choice([3, 5, 8, 12])
    maxd = rng.choice([3, 6, 10, 16, 25])
    lines = [str(n)]
    for _ in range(n):
        t = rng.randint(1, maxt)
        d = rng.randint(1, maxd)
        lines.append(f"{t} {d}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
