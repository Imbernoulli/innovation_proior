# TIER: greedy
"""
The classic Young/Daly recipe: estimate a single GLOBAL mean-time-between-failures from
the whole failure log, plug it into the closed-form optimal interval T* = sqrt(2*C*MTBF),
and checkpoint at fixed multiples of T* for the entire run. This is provably optimal when
failures are memoryless (a homogeneous Poisson process) -- and blind to any local
clustering: a single bad node crashing repeatedly gets exactly the same wide spacing as a
long calm stretch, because the formula only ever sees the global average.
"""
import math
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    W = int(next(it))
    C = int(next(it))
    _R = int(next(it))
    m = int(next(it))
    gaps = [int(next(it)) for _ in range(m)]

    mtbf = (sum(gaps) / m) if m > 0 else float(W)
    mtbf = max(mtbf, 1.0)

    T = max(1, int(round(math.sqrt(2.0 * C * mtbf))))
    thresholds = []
    p = T
    while p < W:
        thresholds.append(p)
        p += T

    print(len(thresholds))
    print(" ".join(map(str, thresholds)))


if __name__ == "__main__":
    main()
