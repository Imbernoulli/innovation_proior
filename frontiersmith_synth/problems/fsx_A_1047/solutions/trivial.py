# TIER: trivial
# Keep only the Urtext (root, node 1) as a full exemplar; every other
# recension ascends straight to its source, all the way to the root.
# This reproduces the checker's own internal baseline exactly (Ratio ~ 0.1).
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    s_budget = int(next(it))
    next(it); next(it)  # size_1, w_1 (unused)
    parent = [0] * (n + 1)
    for i in range(2, n + 1):
        parent[i] = int(next(it))
        next(it); next(it); next(it); next(it)  # up, down, size, w (unused)

    out = [str(1), "1"]
    for v in range(2, n + 1):
        out.append(f"{v} {parent[v]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
