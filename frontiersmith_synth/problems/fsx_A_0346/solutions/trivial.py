# TIER: trivial
# Reproduce the checker's corridor (arrow) baseline: column 0 all +1, row 0 all +1,
# cell (i,i)=-1 for i>=1. |det| = 2^(N-1), the minimum non-zero determinant.
# Scores Ratio = 0.1 on every test.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    rows = []
    for i in range(n):
        row = [1] * n
        if i >= 1:
            row[i] = -1
        rows.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(rows) + "\n")

if __name__ == "__main__":
    main()
