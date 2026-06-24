import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 7)            # small node count
    s = rng.randint(1, n)            # valid source

    # Edge set: simple undirected graph, no self-loops, no duplicate pairs.
    possible = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            possible.append((u, v))
    rng.shuffle(possible)
    # choose how many edges; often sparse so some nodes are unreachable
    max_m = len(possible)
    if max_m == 0:
        m = 0
    else:
        m = rng.randint(0, max_m)
    chosen = possible[:m]

    # brightness values: heavy emphasis on negatives and zeros, small magnitude.
    # Occasionally force an entirely-negative graph.
    mode = rng.randint(0, 3)
    weights = []
    for _ in range(n):
        if mode == 0:
            # all-negative regime
            weights.append(rng.randint(-9, -1))
        elif mode == 1:
            # negatives and zeros only
            weights.append(rng.randint(-9, 0))
        elif mode == 2:
            # mixed including positives, small range
            weights.append(rng.randint(-5, 5))
        else:
            # wider mix with some big magnitudes
            weights.append(rng.choice([rng.randint(-1000, 1000), 0, 0]))

    out = []
    out.append(f"{n} {len(chosen)} {s}")
    out.append(" ".join(str(x) for x in weights))
    for (u, v) in chosen:
        # randomly flip orientation in input to exercise symmetry
        if rng.random() < 0.5:
            out.append(f"{u} {v}")
        else:
            out.append(f"{v} {u}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
