# TIER: invalid
# Emits an out-of-range / non-adjacent artifact -> checker must score 0.
import sys


def main():
    dat = sys.stdin.read().split()
    N = int(dat[0]); K = int(dat[2])
    lines = []
    for i in range(K):
        # single "move" to a node id that does not exist -> infeasible
        lines.append("%d 1 L %d" % (i, N + 7))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
