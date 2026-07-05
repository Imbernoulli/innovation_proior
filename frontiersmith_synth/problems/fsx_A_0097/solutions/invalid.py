# TIER: invalid
# Emits an out-of-range offset (and an oversized count) -> checker rejects.
import sys


def main():
    d = sys.stdin.read().split()
    k = int(d[0]); V = int(d[1])
    A = list(range(k)) + [V + 100]  # last offset exceeds V; also k+1 > k fittings
    print(len(A))
    for x in A:
        print(x)


main()
