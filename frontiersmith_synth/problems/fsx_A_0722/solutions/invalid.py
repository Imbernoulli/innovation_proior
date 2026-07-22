# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    N = int(header[0])
    l_tok = data[1].split()
    K_L = int(l_tok[0])
    # out-of-range index on slot 1 (and too few tokens overall) -> must score 0
    out = [str(K_L + 5)] + ["0"] * (N - 2)
    print(" ".join(out))


main()
