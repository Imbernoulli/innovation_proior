# TIER: invalid
# Emits two full-arena zones stacked on the same point: they overlap grossly,
# so the checker's non-overlap gate rejects the artifact -> Ratio 0.
import sys


def main():
    t = sys.stdin.read().split()
    N = int(t[0]); R = float(t[1])
    lines = ["2", "0.0 0.0 %.10f" % R, "0.0 0.0 %.10f" % R]
    sys.stdout.write("\n".join(lines) + "\n")


main()
