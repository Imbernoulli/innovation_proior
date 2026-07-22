# TIER: invalid
# Every task on arm 0 starting at tick 0: violates arm capacity (and almost always sector
# mutex and precedence too) for any N > 1. Must score 0.0 on every test.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))

    out = []
    for _ in range(N):
        out.append("0 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
