# TIER: trivial
# Do nothing: claim the tool never wears at all. The recursion "0" keeps
# Wprev == 0 forever, so every predicted processing time is just BASE[mat] --
# this reproduces the checker's own no-wear baseline exactly (Ratio == 0.1).
import sys


def main():
    sys.stdin.read()  # consume input (unused)
    print("0")


if __name__ == "__main__":
    main()
