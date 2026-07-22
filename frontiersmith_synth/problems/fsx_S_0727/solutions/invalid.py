# TIER: invalid
# Emits a program that references a register that does not exist yet at the
# time of the very first op (a forward reference), violating the monotone
# append-only ratchet. Must score 0.
import sys


def main():
    sys.stdin.read()  # consume input, ignored
    out = ["3", "0 7", "1 1", "2 9"]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
