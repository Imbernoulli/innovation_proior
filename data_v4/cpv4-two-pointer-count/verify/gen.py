import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 8)
    maxv = random.choice([1, 2, 3, 5, 10])
    a = [random.randint(1, maxv) for _ in range(n)]

    total = sum(a) if a else 0
    hi = max(1, total + 2)

    # Contract guarantees 1 <= L <= R. Pick two values and order them.
    x = random.randint(1, hi)
    y = random.randint(1, hi)
    L, R = min(x, y), max(x, y)

    # Occasionally force tricky boundaries.
    r = random.random()
    if r < 0.15:
        L = 1
    elif r < 0.30 and total > 0:
        L = total
        R = total            # query the single full-array sum exactly
    elif r < 0.45:
        R = L                # exact-value query (L == R)
    elif r < 0.55:
        L = R + 1 if R + 1 <= hi else R  # window just above many sums
        if L > R:
            L, R = R, L

    # Always emit exactly three tokens on the first line.
    print(n, L, R)
    # Second line: the array (empty line if n == 0).
    print(' '.join(map(str, a)))

if __name__ == "__main__":
    main()
