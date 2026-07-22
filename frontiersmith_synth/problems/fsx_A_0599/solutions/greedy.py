# TIER: greedy
# The obvious approach: fit each target ratio in NOMINAL value with the fewest parts --
# one distinct catalog value on top, one on bottom (count 1 each). Nails the nominal
# target but the two DIFFERENT values carry independent corner signs, so worst-corner
# deviation is large. This is the trap.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    K = int(next(it)); M = int(next(it)); P = int(next(it))
    C = int(next(it)); TPM = int(next(it))
    targets = [float(next(it)) for _ in range(K)]
    catalog = [int(next(it)) for _ in range(M)]

    lines = []
    for k in range(K):
        r = targets[k]
        best = None
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                # top = value[i], bottom = value[j]; ratio = vj/(vi+vj)
                ratio = catalog[j] / (catalog[i] + catalog[j])
                e = abs(ratio - r)
                if best is None or e < best[0]:
                    best = (e, i, j)
        _, i, j = best
        top = [0] * M; bot = [0] * M
        top[i] += 1; bot[j] += 1
        lines.append(" ".join(map(str, top + bot)))
    sys.stdout.write("\n".join(lines) + "\n")

main()
