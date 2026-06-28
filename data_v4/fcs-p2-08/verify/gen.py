import random
import sys

def gen(seed):
    rng = random.Random(seed)
    mode = rng.randint(0, 5)

    if mode == 0:
        # Tiny / edge cases.
        n = rng.randint(1, 4)
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        pos = {perm[i]: i for i in range(n)}
        edges = []
        for u in range(1, n + 1):
            for v in range(1, n + 1):
                if pos[u] < pos[v] and rng.random() < 0.5:
                    edges.append((u, v))
        return n, edges

    if mode == 1:
        # Sparse random DAG via random topological labelling.
        n = rng.randint(1, 9)
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        pos = {perm[i]: i for i in range(n)}
        edges = []
        for u in range(1, n + 1):
            for v in range(1, n + 1):
                if pos[u] < pos[v] and rng.random() < 0.3:
                    edges.append((u, v))
        return n, edges

    if mode == 2:
        # Denser random DAG.
        n = rng.randint(2, 8)
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        pos = {perm[i]: i for i in range(n)}
        edges = []
        for u in range(1, n + 1):
            for v in range(1, n + 1):
                if pos[u] < pos[v] and rng.random() < 0.7:
                    edges.append((u, v))
        return n, edges

    if mode == 3:
        # No edges at all (every vertex a source, answer 0).
        n = rng.randint(1, 9)
        return n, []

    if mode == 4:
        # A "wide hub then short tail" shape that tempts the high-out-degree
        # greedy: a hub points to many leaves and also to one node that starts a
        # long chain. Greedy chasing out-degree dives into the leaf fan-out.
        n = rng.randint(5, 9)
        edges = []
        hub = 1
        # hub -> many leaves
        leaves = list(range(2, n))
        for lf in leaves:
            edges.append((hub, lf))
        # hub -> a chain that is long: 2 already used as leaf? rebuild as chain
        # Build a chain among the highest labels.
        chain = list(range(2, n + 1))
        rng.shuffle(chain)
        # ensure topo: relabel chain by sorted order so edges go forward
        chain.sort()
        prev = hub
        for c in chain:
            if rng.random() < 0.6:
                edges.append((prev, c))
                prev = c
        # dedup
        edges = list(set(edges))
        return n, edges

    # mode == 5: layered DAG (multiple sources, multiple sinks).
    layers = rng.randint(2, 4)
    sizes = [rng.randint(1, 3) for _ in range(layers)]
    n = sum(sizes)
    # assign vertex ids per layer in increasing order so edges go forward
    ids = []
    nxt = 1
    for s in sizes:
        layer = list(range(nxt, nxt + s))
        ids.append(layer)
        nxt += s
    edges = []
    for li in range(layers - 1):
        for u in ids[li]:
            for v in ids[li + 1]:
                if rng.random() < 0.6:
                    edges.append((u, v))
            # occasionally skip a layer
            if li + 2 < layers:
                for v in ids[li + 2]:
                    if rng.random() < 0.2:
                        edges.append((u, v))
    return n, edges

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n, edges = gen(seed)
    rng = random.Random(seed * 7919 + 13)
    rng.shuffle(edges)
    out = [f"{n} {len(edges)}"]
    for (u, v) in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
