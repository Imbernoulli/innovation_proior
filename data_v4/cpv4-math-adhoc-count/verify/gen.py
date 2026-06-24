import sys
import random

def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)
    # Mix of tiny and small n so brute force is fast but corner cases appear.
    bucket = seed % 4
    if bucket == 0:
        n = rng.randint(0, 3)
    elif bucket == 1:
        n = rng.randint(1, 30)
    elif bucket == 2:
        n = rng.randint(1, 200)
    else:
        n = rng.randint(1, 600)
    print(n)

if __name__ == "__main__":
    main()
