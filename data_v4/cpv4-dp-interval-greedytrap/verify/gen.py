import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # tiny cases so brute (2^(n-1)) stays cheap
    n = rng.randint(1, 12)
    # Choose W modestly so lines hold a few beads, making the greedy trap reachable.
    W = rng.randint(3, 14)
    # Guarantee feasibility: every bead must fit on a line by itself => w[i] <= W.
    w = [rng.randint(1, W) for _ in range(n)]

    out = []
    out.append(f"{n} {W}")
    out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
