import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    # Small cases so the brute-force DP is cheap and covers the corners:
    #   n = 0 (no colors), k = 0 (empty multiset), c = 0 (no color usable),
    #   and the over/under-fill boundaries of the inclusion-exclusion.
    n = random.randint(0, 8)
    k = random.randint(0, 14)
    c = random.randint(0, 6)
    print(n, k, c)

main()
