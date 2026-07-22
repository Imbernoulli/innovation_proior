# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # every chime gets the SAME address: violates uniqueness and prefix-freeness at once
    out = ["0"] * N
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
