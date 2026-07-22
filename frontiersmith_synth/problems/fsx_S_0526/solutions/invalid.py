# TIER: invalid
# Emits a wire index outside the [0, n+a) range -> strict schema rejection -> 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("CNOT 99999 0\nNOT 0\n")


if __name__ == "__main__":
    main()
