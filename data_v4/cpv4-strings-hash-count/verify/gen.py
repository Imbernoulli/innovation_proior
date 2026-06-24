import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Small cases to stress dedup / off-by-one / k boundaries.
    n = rng.randint(0, 14)
    # alphabet size: sometimes tiny so collisions/repeats are frequent
    asize = rng.choice([1, 2, 2, 3, 4])
    alpha = "abcdefghij"[:asize]
    s = "".join(rng.choice(alpha) for _ in range(n))
    # k chosen to include 0, > n, == n, and typical in-range values
    kchoice = rng.randint(0, n + 2)
    print(n, kchoice)
    print(s)

main()
