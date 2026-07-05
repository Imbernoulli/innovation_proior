# TIER: strong
# Greedy Mian-Chowla Sidon packing (all pairwise sums distinct <=> all pairwise differences
# distinct), skipping reserved weights and staying within [0, M]. A Sidon set of size n has
# |A+A| = n(n+1)/2 and |A-A| = n(n-1)+1, giving quality -> ~2, roughly ten times the AP
# baseline's advantage. Beating 2 (true difference-dominance) is left open.
import sys


def main():
    toks = sys.stdin.read().split()
    n, M, k = int(toks[0]), int(toks[1]), int(toks[2])
    forb = set(int(x) for x in toks[3:3 + k])

    A = []
    sums = set()
    c = 0
    while len(A) < n and c <= M:
        if c in forb:
            c += 1
            continue
        ok = True
        for a in A:
            if a + c in sums:
                ok = False
                break
        if ok:
            for a in A:
                sums.add(a + c)
            sums.add(2 * c)
            A.append(c)
        c += 1

    # safety fallback should the interval be exhausted: fill any remaining distinct legal weights
    c = 0
    used = set(A)
    while len(A) < n and c <= M:
        if c not in forb and c not in used:
            A.append(c)
            used.add(c)
        c += 1

    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
