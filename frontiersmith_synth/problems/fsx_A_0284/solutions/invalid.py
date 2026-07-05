# TIER: invalid
# Emits N towers well outside the plot -> fails containment -> scores 0.
import sys


def main():
    N = int(sys.stdin.read().split()[0])
    out = ["%.6f %.6f" % (5.0 + k, -3.0 - k) for k in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


main()
