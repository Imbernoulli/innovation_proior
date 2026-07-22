# TIER: greedy
"""The obvious textbook approach: start from a filler pattern, then for each
requested border b (processed in the order given in the input -- no attempt to
detect interactions between targets), literally FORCE prefix[0:b] == suffix.
This 'naive stitching' ignores that border constraints are not freely
composable (Fine-Wilf interactions can force extra periods, and overlapping
stitches can silently destroy an earlier border), so it typically both misses
some targets and leaks many more spurious borders than necessary."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    idx += 1  # lam (unused by this naive approach)
    idx += 1  # alpha (unused by this naive approach)
    m = int(toks[idx]); idx += 1
    targets = []
    for _ in range(m):
        b = int(toks[idx]); idx += 1
        w = int(toks[idx]); idx += 1
        targets.append((b, w))

    # start from a simple filler pattern (a non-insightful first move)
    W = [i % K for i in range(n)]

    # stitch each target border directly, in the order it appears in the input
    for b, w in targets:
        for j in range(b):
            W[n - b + j] = W[j]

    sys.stdout.write(" ".join(map(str, W)) + "\n")


if __name__ == "__main__":
    main()
