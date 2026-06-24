import sys
import random

# Larger stress generator: bigger n, bigger values (still small enough for O(n^2) brute),
# and L,R chosen across the full sum range to hit window boundaries and exact-sum queries.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 60)
    maxv = random.choice([1, 2, 1000, 1000000, 1000000000])
    a = [random.randint(1, maxv) for _ in range(n)]

    total = sum(a) if a else 0
    hi = max(1, total + 2)

    x = random.randint(1, hi)
    y = random.randint(1, hi)
    L, R = min(x, y), max(x, y)

    r = random.random()
    if r < 0.2 and total > 0:
        L = R = total
    elif r < 0.4:
        R = L
    elif r < 0.5:
        L = R = hi  # window above everything -> answer 0 often

    assert 1 <= L <= R, (L, R)
    print(n, L, R)
    print(' '.join(map(str, a)))

if __name__ == "__main__":
    main()
