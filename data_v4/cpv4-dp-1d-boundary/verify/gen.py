import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases for brute-force comparison. We deliberately let R sometimes
    # exceed n so that infeasible instances (answer -1) are generated too, and
    # we keep L, R tight so the boundary conditions are exercised.
    n = rng.randint(0, 9)
    L = rng.randint(1, max(1, n) + 1)
    R = rng.randint(L, min(50, max(L, n) + 1))
    K = rng.randint(0, 5)
    vals = [rng.randint(-4, 4) for _ in range(n)]

    out = []
    out.append(f"{n} {K} {L} {R}")
    out.append(" ".join(map(str, vals)))
    sys.stdout.write("\n".join(out) + "\n")

main()
