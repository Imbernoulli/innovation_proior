# TIER: trivial
# Do-nothing baseline: submit the k basis rows with the smallest squared norms.
# This reproduces the checker's internal baseline -> Ratio ~ 0.1.
import sys


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
    B.sort(key=lambda r: sum(x * x for x in r))
    out = []
    for i in range(k):
        out.append(" ".join(str(x) for x in B[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
