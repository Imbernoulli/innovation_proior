# TIER: invalid
"""Emits an infeasible artifact: every state gets the SAME codeword (not a bijection),
which must be rejected by the checker (Ratio: 0.0)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    B = N.bit_length() - 1
    zero = "0" * B
    sys.stdout.write("\n".join([zero] * N) + "\n")


if __name__ == "__main__":
    main()
