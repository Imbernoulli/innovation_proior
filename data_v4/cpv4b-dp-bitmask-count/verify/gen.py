import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(0, 8)          # small for brute force (set partitions blow up fast)
    # L, R chosen so feasibility varies
    L = random.randint(1, max(1, n))
    R = random.randint(L, max(L, n if n > 0 else 1))

    # random feud edges
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(pairs)
    if pairs:
        m = random.randint(0, len(pairs))
    else:
        m = 0
    edges = pairs[:m]

    out = []
    out.append(f"{n} {L} {R} {m}")
    for (u, v) in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")

main()
