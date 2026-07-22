# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    D = int(next(it)); K = int(next(it)); M = int(next(it))
    minlen = int(next(it)); maxlen = int(next(it))
    docs = [next(it) for _ in range(D)]
    # do nothing: ship the empty dictionary (reproduces the checker's own baseline)
    print(0)


if __name__ == "__main__":
    main()
