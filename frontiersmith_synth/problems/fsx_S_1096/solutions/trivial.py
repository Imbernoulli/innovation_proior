# TIER: trivial
# The monomial basis 1, x, x^2, ..., x^{k-1} -- i.e. M = identity. This is exactly
# the checker's own internal baseline construction, so it reproduces Ratio ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    p = int(data[0]); k = int(data[1])
    rows = []
    for i in range(k):
        row = [0] * k
        row[i] = 1
        rows.append(row)
    out = []
    for row in rows:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
