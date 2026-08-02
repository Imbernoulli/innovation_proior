# TIER: invalid
import sys


def main():
    sys.stdin.read()
    # Uses an opcode outside the fixed instruction set -> structurally
    # rejected by the checker's parser before any simulation happens.
    print("FOO r10 r4 1")
    print("RESULT r10 r0 r1 r2 r3")


if __name__ == "__main__":
    main()
