import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the O(V^2) brute force stays fast.
    # Mix structural variety: small V (dense collisions) and larger-V sparse arrays.
    mode = rng.randint(0, 3)
    if mode == 0:
        # tiny everything
        V = rng.randint(0, 6)
        n = rng.randint(0, 8)
    elif mode == 1:
        # small V, many elements -> heavy frequency stacking
        V = rng.randint(1, 12)
        n = rng.randint(1, 30)
    elif mode == 2:
        # larger V, sparse
        V = rng.randint(5, 40)
        n = rng.randint(0, 15)
    else:
        # edge-ish: V can be 0 (all zeros) or n can be 0
        V = rng.randint(0, 3)
        n = rng.randint(0, 5)

    a = [rng.randint(0, V) for _ in range(n)]

    q = rng.randint(0, 10)
    # Queries span the full legal range [0, 3V]; include some boundary values.
    queries = []
    for _ in range(q):
        r = rng.random()
        if r < 0.15:
            queries.append(0)
        elif r < 0.30:
            queries.append(3 * V)
        else:
            queries.append(rng.randint(0, 3 * V))

    lines = []
    lines.append(f"{n} {V}")
    lines.append(" ".join(map(str, a)))
    lines.append(str(q))
    lines.append(" ".join(map(str, queries)))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
