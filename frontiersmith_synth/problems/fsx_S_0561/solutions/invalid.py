# TIER: invalid
# Emits a circuit that does NOT compute the target function (it just outputs x0),
# so the equivalence check fails -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    sys.stdout.write("%d 0\nOUTPUT 0\n" % n)


if __name__ == '__main__':
    main()
