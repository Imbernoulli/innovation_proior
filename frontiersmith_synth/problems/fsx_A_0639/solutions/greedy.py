# TIER: greedy
# The obvious first idea: parts are worth different amounts, so install the
# most valuable part first, then the next most valuable, etc. -- a single
# greedy pass with no lookahead into how today's insertion forecloses
# tomorrow's access corridors.
import sys


def main():
    d = sys.stdin.read().split()
    p = 0
    n = int(d[p]); p += 1
    W = int(d[p]); p += 1
    H = int(d[p]); p += 1
    values = []
    for _ in range(n):
        val = int(d[p]); p += 1
        p += 8  # skip footprint(4) + corridor(4); greedy ignores geometry
        values.append(val)

    order = sorted(range(n), key=lambda i: (-values[i], i))
    print(" ".join(str(i) for i in order))


if __name__ == "__main__":
    main()
