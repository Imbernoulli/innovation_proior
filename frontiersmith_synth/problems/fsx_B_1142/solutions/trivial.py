# TIER: trivial
"""Half-cap baseline: ignore the targets and the topology entirely, set every
edge weight to floor(cap/2). This is exactly the checker's own internal
baseline construction, so it scores ~0.1 by design."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    n = int(toks[p]); p += 1
    m = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    caps = []
    for _ in range(m):
        u = int(toks[p]); v = int(toks[p + 1]); cap = int(toks[p + 2]); p += 3
        caps.append(cap)
    # remaining tokens (times, targets) are not needed by this baseline
    out = [str(max(1, cap // 2)) for cap in caps]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
