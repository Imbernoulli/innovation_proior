import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)
    # node 0 is the root; node i (>=1) attaches to a uniformly random earlier node,
    # which guarantees a valid rooted tree with no cycles.
    par = [-1] * n
    for i in range(1, n):
        par[i] = rng.randint(0, i - 1)

    # small powers so inclusive/exclusive boundaries actually matter
    pw = [rng.randint(0, 3) for _ in range(n)]

    # occasionally force a chain (worst boundary structure) or a star
    shape = rng.randint(0, 2)
    if shape == 1 and n >= 2:
        # chain 0-1-2-...-(n-1)
        for i in range(1, n):
            par[i] = i - 1
    elif shape == 2 and n >= 2:
        # star: everything attaches to root
        for i in range(1, n):
            par[i] = 0

    lines = [str(n)]
    for i in range(n):
        lines.append(f"{par[i]} {pw[i]}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
