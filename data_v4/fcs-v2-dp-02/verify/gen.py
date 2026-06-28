#!/usr/bin/env python3
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small B so the brute force (O(n*q)) and the exact O(3^B) reference stay fast.
    B = rng.randint(0, 12)
    SZ = 1 << B
    n = rng.randint(0, 60)
    q = rng.randint(1, 40)

    items = [rng.randrange(SZ) for _ in range(n)]
    # Mix random queries with some that are exact item masks / full mask / empty.
    queries = []
    for _ in range(q):
        r = rng.random()
        if r < 0.2 and n > 0:
            queries.append(items[rng.randrange(n)])
        elif r < 0.3:
            queries.append(0)
        elif r < 0.4:
            queries.append(SZ - 1)
        else:
            queries.append(rng.randrange(SZ))

    out = [f"{B} {n} {q}"]
    out.append(" ".join(map(str, items)))
    out.append(" ".join(map(str, queries)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
