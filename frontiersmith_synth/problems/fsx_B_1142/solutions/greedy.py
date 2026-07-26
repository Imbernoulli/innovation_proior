# TIER: greedy
"""The obvious first approach: a single shared scale factor applied to every
pipe uniformly (w_e = round(beta * cap_e)), searched over a fine grid of
beta to best match the FULL time grid at once with exact eigen-recomputation.
This is a natural, well-tuned baseline -- but it is fundamentally a ONE
parameter fit. It cannot move the short-time (moment/total-weight) regime
and the long-time (spectral-gap) regime independently, because scaling every
pipe together moves both handles in the same direction at once. When the
hidden target wants total weight and the spectral gap to sit at very
different relative levels, no single beta reaches a good compromise."""
import sys
import math
import numpy as np


def read_instance():
    toks = sys.stdin.read().split()
    p = 0
    n = int(toks[p]); p += 1
    m = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    edges = []
    for _ in range(m):
        u = int(toks[p]); v = int(toks[p + 1]); cap = int(toks[p + 2]); p += 3
        edges.append((u, v, cap))
    ts = [float(toks[p + j]) for j in range(T)]; p += T
    targets = [float(toks[p + j]) for j in range(T)]; p += T
    return n, edges, ts, targets


def trace_values(n, edges, weights, ts):
    L = np.zeros((n, n))
    for (u, v, cap), w in zip(edges, weights):
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    eigs = np.linalg.eigvalsh(L)
    return [float(np.sum(np.exp(-t * eigs))) for t in ts]


def max_log_err(n, edges, weights, ts, targets):
    tv = trace_values(n, edges, weights, ts)
    return max(abs(math.log(max(v, 1e-12)) - math.log(max(g, 1e-12)))
               for v, g in zip(tv, targets))


def main():
    n, edges, ts, targets = read_instance()

    best_w, best_err = None, float("inf")
    for bi in range(0, 101):
        beta = bi / 100.0
        w = [max(1, min(cap, round(beta * cap))) for (u, v, cap) in edges]
        e = max_log_err(n, edges, w, ts, targets)
        if e < best_err:
            best_err, best_w = e, w

    sys.stdout.write(" ".join(str(x) for x in best_w) + "\n")


if __name__ == "__main__":
    main()
