import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # tiny cases so the brute simple-path DFS stays fast
    n = random.randint(1, 6)
    max_edges = n * (n + 1) // 2 + 2
    m = random.randint(0, min(max_edges, 9))
    s = random.randint(1, n)

    # weights span negatives, zero, positives (the whole point of the twist)
    lo, hi = -5, 5
    # occasionally force an all-negative graph
    mode = random.random()
    if mode < 0.2:
        lo, hi = -5, -1   # all negative
    elif mode < 0.35:
        lo, hi = 0, 0     # all zero
    elif mode < 0.5:
        lo, hi = -5, 0    # negatives and zeros only

    lines = [f"{n} {m} {s}"]
    for _ in range(m):
        u = random.randint(1, n)
        v = random.randint(1, n)
        w = random.randint(lo, hi)
        lines.append(f"{u} {v} {w}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
