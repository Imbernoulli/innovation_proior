# TIER: greedy
"""The obvious 'recipe' approach: build the design one session (row) at a time,
never revisiting earlier rows. For each new row, load the k references used
LEAST so far (spread the load), sign each pick to cancel that reference's
running signed total, then run a short bounded local-swap repair against the
rows already placed. This is exactly the single-pass row-by-row
orthogonalization + light polish an average strong coder writes first -- it
has no notion of a global algebraic pattern (circulant structure, quadratic
residues, block tiling), so on instances that hide such structure inside one
n x n grid it plateaus well above the achievable defect."""
import sys
import random
import numpy as np


def main():
    n, k = map(int, sys.stdin.read().split()[:2])
    rnd = random.Random((n * 1000003 + k * 97 + 12345) & 0xFFFFFFFF)

    W = np.zeros((n, n), dtype=np.int64)
    colload = np.zeros(n, dtype=np.int64)
    colsum = np.zeros(n, dtype=np.int64)

    for i in range(n):
        order = sorted(range(n), key=lambda c: (int(colload[c]), c))
        support = order[:k]
        row = np.zeros(n, dtype=np.int64)
        for c in support:
            if colsum[c] > 0:
                sgn = -1
            elif colsum[c] < 0:
                sgn = 1
            else:
                sgn = rnd.choice([1, -1])
            row[c] = sgn

        placed = W[:i]

        def cost(r):
            if i == 0:
                return 0
            return int(np.abs(placed @ r).sum())

        cur = row.copy()
        curc = cost(cur)
        for _ in range(15):
            nz = np.nonzero(cur)[0]
            zeros = np.nonzero(cur == 0)[0]
            if len(nz) == 0 or len(zeros) == 0:
                break
            pos = int(rnd.choice(list(nz)))
            newpos = int(rnd.choice(list(zeros)))
            cand = cur.copy()
            cand[newpos] = cand[pos]
            cand[pos] = 0
            cc = cost(cand)
            if cc < curc:
                cur, curc = cand, cc

        W[i] = cur
        colload += cur * cur
        colsum += cur

    lines = ["\n".join(" ".join(map(str, row)) for row in W.tolist())]
    sys.stdout.write(lines[0] + "\n")


if __name__ == "__main__":
    main()
