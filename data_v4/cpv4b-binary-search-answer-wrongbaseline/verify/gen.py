import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # small cases for brute force (combinations)
    n = rng.randint(2, 9)
    k = rng.randint(2, n)
    L = rng.randint(n + 1, 25)
    # choose n distinct positions in [0, L)
    positions = rng.sample(range(L), n)
    positions.sort()
    out = []
    out.append(f"{n} {k} {L}")
    out.append(" ".join(map(str, positions)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
