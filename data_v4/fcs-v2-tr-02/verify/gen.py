import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # small random tree
    n = rng.randint(1, 12)
    # decide a shape: mix of random parent, path-like, star, and bushy
    shape = rng.randint(0, 3)
    # nodes labeled 1..n; pick a random label to be the root, others get a parent
    # among already-placed nodes to guarantee a valid rooted tree.
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    placed = [perm[0]]
    par = {perm[0]: 0}  # root
    for i in range(1, n):
        node = perm[i]
        if shape == 0:        # random parent
            p = rng.choice(placed)
        elif shape == 1:      # path / caterpillar -> parent is the previous placed
            p = placed[-1]
        elif shape == 2:      # star -> everyone hangs off the root
            p = placed[0]
        else:                 # bushy: parent among last few placed
            p = rng.choice(placed[-3:])
        par[node] = p
        placed.append(node)

    # build queries
    q = rng.randint(1, 15)
    # need depth to sometimes ask in-range, sometimes out-of-range
    depth = {}
    # compute depth by walking up
    for v in range(1, n + 1):
        d = 0
        cur = v
        while par[cur] != 0:
            cur = par[cur]
            d += 1
        depth[v] = d

    lines = []
    lines.append(str(n))
    lines.append(" ".join(str(par[v]) for v in range(1, n + 1)))
    lines.append(str(q))
    for _ in range(q):
        v = rng.randint(1, n)
        mode = rng.randint(0, 3)
        if mode == 0:
            k = 0
        elif mode == 1 and depth[v] > 0:
            k = rng.randint(1, depth[v])          # in range
        elif mode == 2:
            k = rng.randint(depth[v] + 1, depth[v] + 5)  # out of range
        else:
            k = rng.randint(0, n + 3)             # arbitrary
        lines.append(f"{v} {k}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
