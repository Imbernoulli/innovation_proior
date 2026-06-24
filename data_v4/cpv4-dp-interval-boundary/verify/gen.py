import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # tiny cases so the order-enumeration brute force stays fast
    n = rng.randint(0, 8)
    print(n)
    if n > 0:
        vals = [str(rng.randint(1, 12)) for _ in range(n)]
        print(" ".join(vals))

main()
