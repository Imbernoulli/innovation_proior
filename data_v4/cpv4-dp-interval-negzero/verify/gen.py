import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # tiny cases so the exponential brute force stays fast
    n = rng.randint(0, 7)
    # bias toward small magnitudes incl. negatives and zeros to hit the pitfall
    vals = []
    for _ in range(n):
        r = rng.random()
        if r < 0.2:
            vals.append(0)
        else:
            vals.append(rng.randint(-5, 5))
    print(n)
    if n:
        print(" ".join(map(str, vals)))

if __name__ == "__main__":
    main()
