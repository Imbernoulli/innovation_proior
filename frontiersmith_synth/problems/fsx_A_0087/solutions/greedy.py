# TIER: greedy
# Deterministic Mian-Chowla / Golomb greedy: scan positions 0..M and add the
# smallest one that keeps ALL pairwise distances distinct. When the strict
# Sidon constraint can no longer be satisfied within [0,M], fill the remaining
# stages with the smallest unused positions.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])

    chosen = [0]
    diffs = set()
    c = 1
    while len(chosen) < n and c <= M:
        nd = set()
        ok = True
        for x in chosen:
            dd = c - x
            if dd in diffs or dd in nd:
                ok = False
                break
            nd.add(dd)
        if ok:
            chosen.append(c)
            diffs |= nd
        c += 1

    if len(chosen) < n:
        used = set(chosen)
        for p in range(M + 1):
            if len(chosen) >= n:
                break
            if p not in used:
                chosen.append(p)
                used.add(p)

    chosen = chosen[:n]
    print(n)
    print("\n".join(map(str, chosen)))

main()
