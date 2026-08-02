# TIER: invalid
# Investigates every alert, ignoring the time budget entirely -- must score 0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it)); N = int(next(it)); K = int(next(it)); C = int(next(it))
    out = [str(N)]
    out.extend(str(a) for a in range(1, N + 1))
    print("\n".join(out))


if __name__ == "__main__":
    main()
