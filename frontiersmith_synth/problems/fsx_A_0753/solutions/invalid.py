# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it))
    K = int(next(it))
    # Emit K copies of the SAME all-zero codeword: right shape, but every
    # codeword collides with every other one (fails the distinctness check).
    row = "0" * L
    out = [row for _ in range(K)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
