# TIER: invalid
# Emits a single all-zero mode: reconstructs the zero tensor, which does NOT match the
# (nonzero) target -> reconstruction mismatch -> score 0.
import sys


def main():
    d = sys.stdin.read().split()
    a = int(d[0]); b = int(d[1]); c = int(d[2])
    vec = [0] * (a + b + c)
    sys.stdout.write("1\n" + " ".join(str(x) for x in vec) + "\n")


if __name__ == "__main__":
    main()
