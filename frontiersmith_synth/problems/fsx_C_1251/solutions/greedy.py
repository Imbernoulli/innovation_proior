# TIER: greedy
"""The obvious recipe: rank candidate instructions by how often their subgraph occurs
(raw frequency), most-frequent first, and fill the encoding-space (K) / area (A)
budgets in that order. This is a perfectly reasonable heuristic when candidates don't
compete for the same code -- but it has no notion of overlap, so a cluster of several
alternative fusions of the SAME code region all look equally attractive (same
frequency) and can consume most of the budget while contributing almost nothing beyond
the first of them."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    K, A = map(int, data[idx].split()); idx += 1
    C = int(data[idx]); idx += 1
    area = [0] * C
    for c in range(C):
        a, s, cst = map(int, data[idx].split()); idx += 1
        area[c] = a
    M = int(data[idx]); idx += 1
    freq = [0] * C
    for m in range(M):
        L, O = map(int, data[idx].split()); idx += 1
        for _ in range(O):
            cid, start = map(int, data[idx].split()); idx += 1
            freq[cid] += 1

    order = sorted(range(C), key=lambda c: (-freq[c], c))
    sel = []
    used_area = 0
    for c in order:
        if len(sel) >= K:
            break
        if used_area + area[c] <= A:
            sel.append(c)
            used_area += area[c]

    out = [str(len(sel)), " ".join(map(str, sel))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
