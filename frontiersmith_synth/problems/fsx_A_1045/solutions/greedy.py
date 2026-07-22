# TIER: greedy
# Textbook recipe for P|prec|Cmax: rank each job by the length of the longest
# remaining chain of NOMINAL durations below it (critical-path / Hu tail length),
# and list-schedule highest-tail-first. This is the standard "pad the critical
# path" heuristic -- it never looks at which subsystem a job belongs to, so it
# cannot see that a nominally-short job might belong to a dangerous subsystem.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    dur = [0] * (N + 1)
    preds = [[] for _ in range(N + 1)]
    succs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        d = int(next(it)); p = int(next(it))
        plist = [int(next(it)) for _ in range(p)]
        dur[i] = d
        preds[i] = plist
        for pj in plist:
            succs[pj].append(i)
    for _ in range(K):
        num = int(next(it)); den = int(next(it)); s = int(next(it))
        for _ in range(s):
            next(it)

    tail = [0] * (N + 1)
    for i in range(N, 0, -1):
        best_succ = max((tail[s] for s in succs[i]), default=0)
        tail[i] = dur[i] + best_succ

    order = sorted(range(1, N + 1), key=lambda j: (-tail[j], j))
    print(" ".join(map(str, order)))


main()
