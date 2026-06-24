import random, sys

def is_prime(n):
    if n < 2: return False
    for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if n % p == 0:
            return n == p
    d = n - 1; r = 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in [2,3,5,7,11,13,17,19,23,29,31,37]:
        x = pow(a, d, n)
        if x == 1 or x == n-1: continue
        for _ in range(r-1):
            x = x*x % n
            if x == n-1: break
        else:
            return False
    return True

def rand_prime_above(lo):
    # smallest prime >= lo (lo is small here)
    n = lo
    while not is_prime(n):
        n += 1
    return n

def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)

    # Generate queries first (tiny grids so the brute DP is cheap), then pick a
    # prime modulus strictly greater than the largest leg length needed -- this is
    # the problem's stated guarantee (MOD prime, MOD > every cx+cy and dx+dy).
    q = rng.randint(1, 6)
    queries = []
    maxN = 0
    for _ in range(q):
        ex = rng.randint(0, 7)
        ey = rng.randint(0, 7)
        cx = rng.randint(0, ex)
        cy = rng.randint(0, ey)
        queries.append((cx, cy, ex, ey))
        maxN = max(maxN, cx + cy, (ex - cx) + (ey - cy))

    # Choose a prime > maxN. Mix tight-fitting small primes (so brute & sol agree on
    # genuinely small numbers) and occasionally a large prime near 2^31 to exercise
    # the overflow-prone modular multiply.
    choice = rng.randint(0, 3)
    if choice == 0:
        MOD = rand_prime_above(maxN + 1)                       # smallest legal prime
    elif choice == 1:
        MOD = rand_prime_above(max(maxN + 1, rng.randint(13, 200)))
    elif choice == 2:
        MOD = rand_prime_above(rng.randint(10**6, 10**6 + 5000))
    else:
        MOD = rand_prime_above(rng.randint(2_000_000_000, 2_000_010_000))  # near 2^31

    lines = [f"{q} {MOD}"]
    for (cx, cy, ex, ey) in queries:
        lines.append(f"{cx} {cy} {ex} {ey}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
