import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # small grids so brute force (repeated relaxation) is fast
    n = rng.randint(1, 6)
    m = rng.randint(1, 6)
    # small height alphabet so ties/glides/boosts all occur frequently
    hi = rng.choice([1, 2, 3, 5, 9])
    out = [f"{n} {m}"]
    for i in range(n):
        row = [str(rng.randint(0, hi)) for _ in range(m)]
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")

main()
