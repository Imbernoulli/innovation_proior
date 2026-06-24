import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    k = rng.randint(0, n)            # 0 <= k <= n, may remove all
    # Small alphabet sometimes, full digits other times, to stress ties.
    alpha = rng.choice(["01", "012", "0123456789", "0123456789"])
    s = "".join(rng.choice(alpha) for _ in range(n))
    print(n, k)
    print(s)

main()
