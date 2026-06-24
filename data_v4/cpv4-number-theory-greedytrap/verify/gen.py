import sys, random

# Small-case generator parameterized by an integer seed.
# Respects the contract: 2 <= k <= 5, 0 <= n <= N_MAX (kept tiny here so the
# BFS brute force stays fast). Includes the n == 0 corner occasionally.

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    k = random.randint(2, 5)
    r = random.random()
    if r < 0.05:
        n = 0
    elif r < 0.15:
        n = random.randint(1, 5)        # very small
    else:
        n = random.randint(1, 400)      # small but big enough to expose greedy
    print(k, n)

main()
