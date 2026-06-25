import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 6)
    maxw = rng.choice([1, 2, 3, 5, 8, 12])
    w = [rng.randint(1, maxw) for _ in range(n)]

    # k chosen across the interesting range, including 0 (forces p >= max weight)
    # and values that land exactly on divisibility boundaries.
    total_pieces_upper = sum(w)  # blows at p=1 = sum(w)-n; pick k around there
    k = rng.randint(0, max(0, sum(w) - n) + 2)

    out = [f"{n} {k}", " ".join(map(str, w))]
    sys.stdout.write("\n".join(out) + "\n")

main()
