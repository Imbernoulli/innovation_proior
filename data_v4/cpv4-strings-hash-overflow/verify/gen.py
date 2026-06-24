import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # small cases; use a small alphabet to force many equal windows (collisions of content)
    n = rng.randint(1, 12)
    L = rng.randint(1, n + 2)  # sometimes L > n to test the empty-window branch
    alpha_size = rng.choice([1, 2, 2, 3])  # small alphabet -> repeated windows
    alphabet = "abcdefghij"[:alpha_size]
    s = "".join(rng.choice(alphabet) for _ in range(n))
    print(n, L)
    print(s)

main()
