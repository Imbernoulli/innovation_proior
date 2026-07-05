# TIER: invalid
# Emits out-of-range coordinates and the wrong number of points -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    d, M = int(toks[0]), int(toks[1])
    # wrong count AND out-of-range values
    out = []
    for i in range(M + 3):
        out.append("%.4f %.4f" % (7.5, -2.0))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
