# TIER: invalid
# Emits N pads well outside the triangular plate -> fails containment -> 0.
import sys


def main():
    N = int(sys.stdin.read().split()[0])
    out = ["%.6f %.6f" % (5.0 + k, 5.0 + k) for k in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


main()
