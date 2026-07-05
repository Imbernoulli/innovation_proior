# TIER: greedy
# Centred regular grid: k = ceil(sqrt(M)) columns/rows, place probes at cell
# centres ((c+0.5)/k,(r+0.5)/k) taking the first M cells in row-major order.
# Uniform but O(1/sqrt(M)) discrepancy -- beats the diagonal, worse than a
# proper low-discrepancy sequence.
import sys
import math


def main():
    t = sys.stdin.read().split()
    d = int(t[0]); m = int(t[1])
    k = int(math.ceil(math.sqrt(m)))
    out = []
    placed = 0
    for r in range(k):
        for c in range(k):
            if placed >= m:
                break
            x = (c + 0.5) / k
            y = (r + 0.5) / k
            out.append("%.10f %.10f" % (x, y))
            placed += 1
        if placed >= m:
            break
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
