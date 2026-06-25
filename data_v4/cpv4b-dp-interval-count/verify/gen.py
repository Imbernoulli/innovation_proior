import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep k^n small enough for the brute force to enumerate.
    # n in [0, 9], k in [1, 4]; cap k^n by retrying.
    while True:
        n = rng.randint(0, 9)
        k = rng.randint(1, 4)
        if k ** max(n, 1) <= 60000:
            break

    # A <= B, both at least 1 (a run has positive length).
    A = rng.randint(1, max(1, n))
    B = rng.randint(A, max(A, n + 1))

    # Modulus: mix small (to exercise reduction) and larger primes.
    M = rng.choice([2, 3, 5, 7, 13, 97, 1000, 998244353, 1000000007])

    print(n, k, A, B, M)

main()
