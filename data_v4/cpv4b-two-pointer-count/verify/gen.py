import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases to stress the window edges, ties, and L/R corners.
    n = rng.randint(0, 9)
    # Small value range so ties and exact-bound hits happen often.
    vmax = rng.choice([1, 2, 3, 5, 8])
    # L, R chosen from a range that includes 0 and values above vmax sometimes.
    L = rng.randint(0, vmax + 1)
    R = rng.randint(L, vmax + 2)

    f = [rng.randint(-vmax, vmax) for _ in range(n)]

    out = []
    out.append(f"{n}")
    out.append(f"{L} {R}")
    out.append(" ".join(map(str, f)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
