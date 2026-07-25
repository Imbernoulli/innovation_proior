# TIER: greedy
# The obvious recipe: sort jobs by duty cycle (d_i/p_i) descending, then place
# each job at the offset that minimizes its EXACT marginal waste against the
# jobs already placed ("earliest cheapest slot", ties -> smallest offset).
# It never reserves the cheap residue class for the jobs that could tile it,
# so the planted high-duty flexible family consumes expensive real estate as
# singles and the late rigid jobs' unavoidable overlap lands on heavy instants.
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    n = int(next(it))
    jobs = [(int(next(it)), int(next(it))) for _ in range(n)]
    w = np.array([float(next(it)) for _ in range(M)], dtype=np.float64)

    load = np.zeros(M, dtype=np.int32)
    order = sorted(range(n), key=lambda i: (-jobs[i][1] / jobs[i][0], jobs[i][0], i))
    offs = [0] * n
    tau = np.arange(0, max(d for (_, d) in jobs))

    for i in order:
        p, d = jobs[i]
        occ = (load >= 1).astype(np.float64)
        col = (w * occ).reshape(M // p, p).sum(axis=0)      # cost per residue
        ext = np.concatenate([col, col[:d - 1]]) if d > 1 else col
        csum = np.concatenate([[0.0], np.cumsum(ext)])
        cost = csum[d:d + p] - csum[:p]                      # cost per offset
        o = int(np.argmin(cost))                             # first (smallest) minimizer
        offs[i] = o
        base = (o + tau[:d]) % p
        idx = (np.arange(M // p, dtype=np.int64)[:, None] * p + base[None, :]).ravel()
        load[idx] += 1

    print(" ".join(map(str, offs)))


if __name__ == "__main__":
    main()
