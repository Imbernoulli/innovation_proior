# TIER: trivial
# Reproduce the checker baseline exactly: the "shoreline ridge" subcube
# (first digit 0, all other digits in {0,1}). Always raid-free, never blocked.
import sys, itertools


def main():
    raw = sys.stdin.read().splitlines()
    n = int(raw[0].split()[0])
    out = []
    for v in itertools.product(range(3), repeat=n):
        s = ''.join(map(str, v))
        if s[0] == '0' and all(c in '01' for c in s[1:]):
            out.append(s)
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
