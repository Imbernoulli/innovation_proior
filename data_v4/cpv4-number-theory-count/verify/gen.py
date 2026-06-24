import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    t = rng.randint(1, 5)
    print(t)
    for _ in range(t):
        # tiny N so brute O(N^2) with a set stays cheap
        n = rng.randint(1, 60)
        print(n)

if __name__ == "__main__":
    main()
