# TIER: greedy
"""Marginal mutual-information thresholding.  Add an (undirected) link between
every pair of subsystems whose empirical mutual information exceeds a fixed
threshold, and orient it by raw column order (i -> j for i < j).  This catches
real direct links but also every INDIRECT (transitive) association along a causal
path as a false edge, and because the subsystems are randomly relabelled the
column-order orientation is essentially a coin flip -- so it recovers some
skeleton yet pays heavily in extra and reversed edges."""
import sys, json
import numpy as np


def mutual_info(x, y, K):
    N = len(x)
    joint = np.zeros((K, K))
    np.add.at(joint, (x, y), 1.0)
    joint /= N
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    m = 0.0
    for a in range(K):
        if px[a] <= 0:
            continue
        for b in range(K):
            if joint[a, b] > 0 and py[b] > 0:
                m += joint[a, b] * np.log(joint[a, b] / (px[a] * py[b]))
    return m


def main():
    inst = json.load(sys.stdin)
    data = np.asarray(inst["data"], dtype=np.int64)
    d = int(inst["n_nodes"])
    K = int(inst.get("n_categories", int(data.max()) + 1))
    thr = 0.06
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if mutual_info(data[:, i], data[:, j], K) > thr:
                edges.append([i, j])          # orient by column order (uninformed)
    print(json.dumps({"edges": edges}))


main()
