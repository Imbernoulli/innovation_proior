# TIER: invalid
# Emits a single bogus stage of all-ones factors that does NOT reconstruct the
# tensor -> exact-equality gate fails -> Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    a = int(data[0]); b = int(data[1]); c = int(data[2])
    row = " ".join(["1"] * (a + b + c))
    sys.stdout.write("1\n" + row + "\n")


if __name__ == "__main__":
    main()
