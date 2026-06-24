import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Keep n small so the brute force (enumerate all compositions) stays fast.
    n = rng.randint(0, 14)
    K = rng.randint(1, 6)
    # Use a mix of small primes / composite moduli to exercise modular arithmetic.
    MOD = rng.choice([2, 3, 5, 7, 11, 13, 97, 1000, 998244353, 1000000007])
    print(n, K, MOD)

if __name__ == "__main__":
    main()
