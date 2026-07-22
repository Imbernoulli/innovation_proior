# TIER: invalid
# Emits an out-of-bounds input (amplitude way past the bound) -> must score 0.
import sys


def main():
    tk = sys.stdin.read().split()
    N = int(tk[0])
    A = float(tk[1])
    sys.stdout.write(" ".join([repr(100.0 * A)] * N) + "\n")


if __name__ == "__main__":
    main()
