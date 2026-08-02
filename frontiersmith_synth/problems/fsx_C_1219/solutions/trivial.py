# TIER: trivial
import sys


def main():
    sys.stdin.read()  # consume the instance (unused: identity strategy)
    # Never adapt: forward the previous window unchanged forever. This is
    # exactly the checker's own internal baseline construction.
    print("RESULT r4 r0 r1 r2 r3")


if __name__ == "__main__":
    main()
