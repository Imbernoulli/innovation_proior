# TIER: trivial
# The textbook default: identity alphabet order, no rotation. The checker's own
# baseline B is the fixed trivial upper bound n+1 (see counter.py), not this
# construction's achieved run count -- this is simply the weakest sensible reference,
# scoring modestly above 0.1 since even the untouched default already compresses T.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    # T values are ignored -- the identity order + r=0 needs no information from T.
    order = list(range(k))
    print(" ".join(map(str, order)))
    print(0)


if __name__ == "__main__":
    main()
