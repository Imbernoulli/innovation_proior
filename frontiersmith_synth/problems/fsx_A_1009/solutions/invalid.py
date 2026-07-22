# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); cap = int(next(it)); K = int(next(it))

    # Claim to use m+3 probes (exceeds the budget m) -- must be rejected outright.
    r = m + 3
    print(r)
    for i in range(r):
        # also violate the probe cap by lighting up every component, for good measure
        print("1" * n)


if __name__ == "__main__":
    main()
