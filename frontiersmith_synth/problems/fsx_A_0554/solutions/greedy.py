# TIER: greedy
# The obvious first attempt: "just make the basis vectors shorter by subtracting integer
# multiples of one another" -- i.e. size-reduction (Babai rounding) on the full basis, with
# NO Lovasz swaps / reordering.  This lowers the norms somewhat but cannot reach the deeply
# short vectors that require a global reordering of the badly-conditioned scrambled basis,
# and it is completely blind to the planted congruence structure.
import sys


def gram_schmidt(B):
    n = len(B)
    m = len(B[0])
    Bs = []
    mu = [[0.0] * n for _ in range(n)]
    for i in range(n):
        bi = [float(x) for x in B[i]]
        for j in range(i):
            denom = sum(x * x for x in Bs[j])
            if denom == 0:
                mu[i][j] = 0.0
                continue
            mu[i][j] = sum(B[i][t] * Bs[j][t] for t in range(m)) / denom
            bi = [bi[t] - mu[i][j] * Bs[j][t] for t in range(m)]
        Bs.append(bi)
    return Bs, mu


def size_reduce(B, rounds=40):
    n = len(B)
    m = len(B[0])
    for _ in range(rounds):
        changed = False
        Bs, mu = gram_schmidt(B)
        for i in range(n):
            for j in range(i - 1, -1, -1):
                q = int(round(mu[i][j]))
                if q != 0:
                    B[i] = [B[i][t] - q * B[j][t] for t in range(m)]
                    changed = True
            if changed:
                Bs, mu = gram_schmidt(B)
        if not changed:
            break
    return B


def independent_pick(cands, k):
    # greedily pick k independent vectors (shortest first) via float Gaussian elimination
    chosen = []
    basis = []  # reduced float rows
    for v in sorted(cands, key=lambda r: sum(x * x for x in r)):
        row = [float(x) for x in v]
        for b in basis:
            # reduce row against b at b's pivot
            pc = b[1]
            if row[pc] != 0:
                f = row[pc] / b[0][pc]
                row = [row[t] - f * b[0][t] for t in range(len(row))]
        pc = next((c for c in range(len(row)) if abs(row[c]) > 1e-7), None)
        if pc is None:
            continue
        basis.append((row, pc))
        chosen.append(v)
        if len(chosen) == k:
            break
    return chosen


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    p = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    B = []
    for i in range(n):
        row = [int(toks[idx + j]) for j in range(n)]
        idx += n
        B.append(row)

    B = size_reduce([r[:] for r in B])
    chosen = independent_pick([r[:] for r in B], k)
    out = []
    for v in chosen:
        out.append(" ".join(str(x) for x in v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
