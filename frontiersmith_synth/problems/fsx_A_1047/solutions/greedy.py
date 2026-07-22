# TIER: greedy
# The "obvious" recipe: always keep the Urtext as a safety exemplar, then
# spend the remaining parchment budget on whichever recensions are requested
# most often (raw pilgrim-demand ranking, ignoring how expensive each one is
# to keep and ignoring the asymmetric ascent/descent labor costs). Every
# non-exemplar recension is always chained UPWARD to its source -- the
# textbook "chain to nearest ancestor" convention -- never reoriented to
# descend toward a cheaper exemplar living in its own subtree.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    s_budget = int(next(it))
    size = [0] * (n + 1)
    w = [0.0] * (n + 1)
    parent = [0] * (n + 1)
    size[1] = int(next(it))
    w[1] = float(next(it))
    for i in range(2, n + 1):
        parent[i] = int(next(it))
        next(it); next(it)  # up, down (ignored by the naive recipe)
        size[i] = int(next(it))
        w[i] = float(next(it))

    checkpoints = [1]
    is_ckpt = [False] * (n + 1)
    is_ckpt[1] = True
    remaining = s_budget - size[1]

    candidates = sorted(range(2, n + 1), key=lambda v: -w[v])
    for v in candidates:
        if size[v] <= remaining:
            is_ckpt[v] = True
            checkpoints.append(v)
            remaining -= size[v]

    out = [str(len(checkpoints)), " ".join(map(str, checkpoints))]
    for v in range(1, n + 1):
        if is_ckpt[v]:
            continue
        out.append(f"{v} {parent[v]}")  # always ascend to the source
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
