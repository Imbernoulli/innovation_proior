# TIER: trivial
# Reproduces the checker's own baseline construction exactly: split the T-point
# budget round-robin across the K families, and within each family aim at the
# midpoints of a few evenly-spaced mutants. Ignores which family is actually
# hardest to discriminate. Scores Ratio = 0.1 on every test by construction.
import sys


def main():
    toks = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = int(toks[pos[0]])
        pos[0] += 1
        return v

    D = nxt()
    T = nxt()
    K = nxt()
    families = []
    for _ in range(K):
        n_f = nxt()
        fam = [(nxt(), nxt()) for _ in range(n_f)]
        families.append(fam)

    alloc = [T // K] * K
    rem = T - sum(alloc)
    for i in range(rem):
        alloc[i] += 1

    xs = []
    for fi, fam in enumerate(families):
        n_f = len(fam)
        a_f = min(alloc[fi], n_f)
        if a_f <= 0:
            continue
        if a_f == 1:
            idxs = [0]
        else:
            idxs = sorted(set(round(i * (n_f - 1) / (a_f - 1)) for i in range(a_f)))
        for j in idxs:
            lo, hi = fam[j]
            xs.append((lo + hi - 1) // 2)

    xs = xs[:T]
    print(len(xs))
    print(" ".join(map(str, xs)))


if __name__ == "__main__":
    main()
