# TIER: trivial
# Process jobs in the given input order with no re-dressing. This reproduces the
# checker's internal baseline construction -> scores ~0.1 by design.
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    sys.stdout.write(" ".join(str(i) for i in range(1, N + 1)))


if __name__ == "__main__":
    main()
