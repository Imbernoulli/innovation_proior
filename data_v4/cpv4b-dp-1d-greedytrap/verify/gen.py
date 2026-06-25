import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 9)
    # allow negative costs (currents that push you), small range to make
    # greedy traps frequent and keep brute-force fast
    lo, hi = -6, 9
    vals = [rng.randint(lo, hi) for _ in range(n)]
    print(n)
    if n > 0:
        print(" ".join(map(str, vals)))

if __name__ == "__main__":
    main()
