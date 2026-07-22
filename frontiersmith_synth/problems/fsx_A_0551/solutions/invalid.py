# TIER: invalid
"""Emits an infeasible edit (digs a cell far below sea level -> height < 0),
which the checker must reject with Ratio 0.0."""
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    def nx():
        return int(next(it))
    N = nx()
    # one edit that drives a cell's height hugely negative -> out of [0,HMAX]
    sys.stdout.write("1\n%d %d %d\n" % (N // 2, N // 2, -1000000))


if __name__ == "__main__":
    main()
