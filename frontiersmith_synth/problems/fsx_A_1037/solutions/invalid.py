# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); R = int(next(it)); K = int(next(it))
    # deliberately infeasible: reference a machine id (M+1) that does not exist.
    sys.stdout.write("1\n1 %d\n" % (M + 1))


main()
