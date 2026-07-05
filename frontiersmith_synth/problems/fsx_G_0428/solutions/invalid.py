# TIER: invalid
# Emits a malformed expression referencing a disallowed name -> grader rejects (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("a1 + banana * n\n")


if __name__ == "__main__":
    main()
