# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    k = int(d[0]); V = int(d[1])
    m = min(k, V + 1)
    A = list(range(m))  # contiguous ramp = checker baseline
    print(m)
    for x in A:
        print(x)


main()
