# TIER: invalid
"""Deliberately infeasible: includes the degenerate baseline triple
{0, e_1, 2*e_1} (0 + 1 + 2 == 3 == 0 mod 3 in coord 1, 0 elsewhere), so the
checker must reject it with Ratio 0.0."""
import sys


def main():
    n = int(sys.stdin.readline().split()[0])
    out = []
    out.append(" ".join(["0"] * n))          # 0
    a = ["0"] * n; a[0] = "1"; out.append(" ".join(a))   # e_1
    b = ["0"] * n; b[0] = "2"; out.append(" ".join(b))   # 2*e_1  -> line with the two above
    # pad with more vectors so the set is large but still invalid
    for i in range(1, n):
        v = ["0"] * n; v[i] = "1"; out.append(" ".join(v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
