# TIER: greedy
# The "obvious" recipe: equal-capital allocation across all sleeves -- it LOOKS
# diversified (every sleeve gets the same slice) and is blind to both the calm and the
# stress data entirely. Only projected onto the per-sleeve caps and the cluster group cap
# to stay feasible. This is exactly the textbook trap the innovation hook warns about:
# equal capital concentrates realized tail risk in the highest-volatility sleeves,
# because those are exactly the ones that crash together in the stress regime.
import sys
import numpy as np


def read_instance():
    toks = sys.stdin.read().split()
    pos = 0
    tid = int(toks[pos]); pos += 1
    N = int(toks[pos]); pos += 1
    cap = np.array([float(toks[pos + i]) for i in range(N)]); pos += N
    group_cap = float(toks[pos]); pos += 1
    K = int(toks[pos]); pos += 1
    cluster = [int(toks[pos + i]) for i in range(K)]; pos += K
    c_calm = int(toks[pos]); pos += 1
    pos += c_calm * N
    c_stress = int(toks[pos]); pos += 1
    pos += c_stress * N
    return N, cap, group_cap, cluster


def project(w0, cap, cluster_mask, group_cap, iters=300):
    w = np.clip(w0, 0, cap).astype(float)
    for _ in range(iters):
        csum = w[cluster_mask].sum()
        if csum > group_cap + 1e-9:
            w[cluster_mask] *= group_cap / csum
        w = np.minimum(w, cap)
        total = w.sum()
        if abs(total - 1.0) < 1e-10:
            break
        diff = 1.0 - total
        if diff > 0:
            slack = cap - w
            if w[cluster_mask].sum() >= group_cap - 1e-9:
                slack[cluster_mask] = 0.0
            ssum = slack.sum()
            if ssum <= 1e-12:
                break
            w += diff * slack / ssum
        else:
            pos = w > 1e-12
            psum = w[pos].sum()
            if psum <= 1e-12:
                break
            w[pos] += diff * w[pos] / psum
            w = np.maximum(w, 0.0)
    return w


def main():
    N, cap, group_cap, cluster = read_instance()
    cluster_mask = np.zeros(N, dtype=bool)
    cluster_mask[cluster] = True
    w = project(np.ones(N) / N, cap, cluster_mask, group_cap)
    sys.stdout.write(" ".join("%.8f" % v for v in w) + "\n")


main()
