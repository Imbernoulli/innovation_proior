# TIER: trivial
# Do nothing: neutral feedback exponent mu=1.0, no initial-thickness overrides
# at all (every edge starts at the checker's own default D_BASE). This is
# exactly the checker's internal baseline construction, so it reproduces B.
import sys


def main():
    sys.stdin.read()  # instance is irrelevant to this submission
    sys.stdout.write("1.0\n0\n")


if __name__ == "__main__":
    main()
