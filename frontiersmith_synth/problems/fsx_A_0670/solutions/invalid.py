# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, M, K = (int(x) for x in data[0].split())
    reqs = [int(x) for x in data[2].split()] if K > 0 else []
    if not reqs:
        return
    r1 = reqs[0]
    # Skip the mandatory forward build-up entirely and just try to "serve"
    # the largest request straight away -- violates both the dependency rule
    # and (if r1 == 1) still leaves every later required serve unmet.
    print("C 1")
    print("U %d" % r1)


if __name__ == "__main__":
    main()
