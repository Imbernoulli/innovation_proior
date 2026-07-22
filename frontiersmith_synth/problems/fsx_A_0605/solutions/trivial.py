# TIER: trivial
# Do-nothing: predict next density = current density (persistence).  This is
# exactly the grader's internal baseline, so it reproduces Ratio ~ 0.1.
import sys


def main():
    sys.stdin.read()          # ignore the census
    print("x")


if __name__ == "__main__":
    main()
