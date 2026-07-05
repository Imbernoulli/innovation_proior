# TIER: invalid
# Emits a single all-zero mode: reconstructs the zero tensor != T -> ratio 0.
import sys


def main():
    inp = sys.stdin.read().split()
    I, J, K = int(inp[0]), int(inp[1]), int(inp[2])
    out = ["1", " ".join(["0"] * I), " ".join(["0"] * J), " ".join(["0"] * K)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
