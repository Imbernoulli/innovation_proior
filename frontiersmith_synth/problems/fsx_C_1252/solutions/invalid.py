# TIER: invalid
import sys

# Emits zero buffers on every net -- with a nonzero raw wire-delay spread
# and a nominal-skew budget strictly tighter than that raw spread, this is
# guaranteed infeasible.


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1; return v
    K = int(nxt()); m = int(nxt()); C = int(nxt())
    for _ in range(m):
        nxt(); nxt()
        for _ in range(C):
            nxt()
    nxt(); nxt()
    for _ in range(K):
        nxt()

    out = []
    for i in range(K):
        out.append(" ".join(["0"] * m))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
