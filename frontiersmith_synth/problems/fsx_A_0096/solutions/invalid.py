# TIER: invalid
# Emits a matrix that violates the fixed-first-row constraint (negates r0) and
# uses out-of-alphabet zeros -> feasibility gate rejects it -> Ratio 0.0.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it))
    r0 = [int(next(it)) for _ in range(n)]
    rows = []
    rows.append(" ".join(str(-x) for x in r0))          # wrong first row
    for _ in range(n - 1):
        rows.append(" ".join("0" for _ in range(n)))    # zeros are not +/-1
    sys.stdout.write("\n".join(rows) + "\n")

main()
