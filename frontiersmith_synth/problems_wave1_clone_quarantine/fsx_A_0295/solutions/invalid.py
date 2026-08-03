# TIER: invalid
# Emits negative capacities (physically meaningless). The checker rejects any
# negative value, so this scores 0.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    f = [-1.0] * n
    print(" ".join("%.6f" % x for x in f))


if __name__ == "__main__":
    main()
