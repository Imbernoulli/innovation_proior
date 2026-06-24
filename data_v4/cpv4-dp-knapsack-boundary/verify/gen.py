import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 8)
    # Keep U small so brute force (2^n subsets) and the boundary are exercised.
    K = random.randint(0, 14)
    g = random.randint(0, 14)

    print(n, K, g)
    for _ in range(n):
        si = random.randint(0, 10)
        vi = random.randint(0, 20)
        print(si, vi)

if __name__ == "__main__":
    main()
