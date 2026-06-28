import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    q = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    shape = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 random, 1 path, 2 star, 3 skewed
    rng = random.Random(seed)

    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    par = [0] * (n + 1)
    par[perm[0]] = 0
    placed = [perm[0]]
    for i in range(1, n):
        node = perm[i]
        if shape == 0:
            p = placed[rng.randrange(len(placed))]
        elif shape == 1:
            p = placed[-1]
        elif shape == 2:
            p = placed[0]
        else:
            # skewed: parent close to the most recent => long paths
            j = max(0, len(placed) - rng.randint(1, 3))
            p = placed[j]
        par[node] = p
        placed.append(node)

    # depths for query targeting
    depth = [0] * (n + 1)
    # compute by topological order: placed is already a valid order (parents first)
    for node in placed:
        if par[node] != 0:
            depth[node] = depth[par[node]] + 1

    out = []
    out.append(str(n))
    out.append(" ".join(str(par[v]) for v in range(1, n + 1)))
    out.append(str(q))
    for _ in range(q):
        v = rng.randint(1, n)
        r = rng.random()
        if r < 0.5 and depth[v] > 0:
            k = rng.randint(0, depth[v])
        else:
            k = rng.randint(0, depth[v] + 5)
        out.append(f"{v} {k}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
