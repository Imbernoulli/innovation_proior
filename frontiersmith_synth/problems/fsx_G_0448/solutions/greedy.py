# TIER: greedy
"""Support-pruned schoolbook: only form the products x_i*y_j for pairs (i,j) that
actually appear in SOME output (nonzero column T[:,i,j]).  Skips structurally-absent
products, beating the full schoolbook baseline when the tensor is sparse, but makes
no use of linear-rank structure across products -> a genuine middle strategy."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    p = int(next(it)); q = int(next(it)); r = int(next(it))
    T = [[[0] * q for _ in range(p)] for _ in range(r)]
    for k in range(r):
        for i in range(p):
            for j in range(q):
                T[k][i][j] = int(next(it))

    lines = []
    R = 0
    for i in range(p):
        for j in range(q):
            col = [T[k][i][j] for k in range(r)]
            if all(v == 0 for v in col):
                continue
            a = [0] * p; a[i] = 1
            c = [0] * q; c[j] = 1
            lines.append(" ".join(map(str, a + c + col)))
            R += 1
    if R == 0:  # degenerate all-zero tensor: emit one harmless zero term
        a = [0] * p; c = [0] * q; d = [0] * r
        lines.append(" ".join(map(str, a + c + d)))
        R = 1
    sys.stdout.write(str(R) + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
