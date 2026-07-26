# TIER: invalid
# Sends every ship as an unescorted lease from tick 0, with ZERO escorted trips ever run.
# No cell is ever cleared, so every ship's very first cell check fails -> Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    L = int(data[p]); p += 1
    M = int(data[p]); p += 1

    out = ["0", str(M)]
    for j in range(1, M + 1):
        out.append("%d 0" % j)
    sys.stdout.write("\n".join(out) + "\n")


main()
