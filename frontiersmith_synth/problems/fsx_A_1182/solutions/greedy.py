# TIER: greedy
"""The obvious first idea: 'the best-connected sick person started it.'
Weight every infected candidate by its raw contact-network degree (full
graph, not just infected-induced). This is exactly the trap the innovation
hook warns about: a hub is reachable from everywhere, so it gets infected
early no matter where the outbreak actually started, and raw degree crowns
it regardless."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, M = map(int, data[1].split())
    deg = [0] * N
    ptr = 2
    for _ in range(M):
        u, v, w = map(int, data[ptr].split())
        ptr += 1
        deg[u] += 1
        deg[v] += 1
    T = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    cand = list(map(int, data[ptr].split()))

    weights = [float(deg[c]) + 0.1 for c in cand]  # +0.1: never all-zero

    out = sys.stdout
    out.write("%d\n" % K)
    out.write(" ".join("%.6f" % w for w in weights))
    out.write("\n")


if __name__ == "__main__":
    main()
