# TIER: invalid
# Emits a single self-loop "waggle" (cells 0,0): not a hardware edge and the
# interaction multiset cannot match -> the checker must score this 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("G 0 0\n")


if __name__ == "__main__":
    main()
