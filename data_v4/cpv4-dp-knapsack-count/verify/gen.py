import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the 2^n brute force stays cheap, but with enough structure
    # to trigger the double-count / clamp pitfalls (collisions on price, joy
    # crossing the J threshold, items with price > B, J = 0, etc.).
    n = rng.randint(0, 12)
    B = rng.randint(0, 14)
    J = rng.randint(0, 10)

    items = []
    for _ in range(n):
        p = rng.randint(0, 14)   # prices can equal/exceed B and can be 0
        j = rng.randint(0, 8)    # joy can be 0
        items.append((p, j))

    out = []
    out.append(f"{n} {B} {J}")
    for (p, j) in items:
        out.append(f"{p} {j}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
