# TIER: trivial
# Naive momentum: estimate each sleeve's calm-regime mean return from the visible calm
# sample and chase the single best track record, filling the rest of the book (subject
# to the per-sleeve cap and the cluster group cap) in descending order of that same
# estimate. Uses no stress information at all.
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
    calm = np.array([float(toks[pos + i]) for i in range(c_calm * N)]).reshape(c_calm, N)
    pos += c_calm * N
    c_stress = int(toks[pos]); pos += 1
    stress = np.array([float(toks[pos + i]) for i in range(c_stress * N)]).reshape(c_stress, N)
    pos += c_stress * N
    return N, cap, group_cap, cluster, calm, stress


def main():
    N, cap, group_cap, cluster, calm, stress = read_instance()
    calm_mean = calm.mean(axis=0)
    cluster_set = set(cluster)

    order = np.argsort(-calm_mean)
    w = np.zeros(N)
    remaining = 1.0
    cluster_used = 0.0
    for i in order:
        avail = cap[i]
        if i in cluster_set:
            avail = min(avail, group_cap - cluster_used)
        take = max(min(avail, remaining), 0.0)
        w[i] = take
        if i in cluster_set:
            cluster_used += take
        remaining -= take
        if remaining <= 1e-12:
            break

    sys.stdout.write(" ".join("%.8f" % v for v in w) + "\n")


main()
