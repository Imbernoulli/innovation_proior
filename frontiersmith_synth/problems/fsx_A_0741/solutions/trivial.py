# TIER: trivial
"""Never merge anything: every box stays its own block forever. Reproduces
the checker's worst-case read exposure -- correct but leaves every overlap
unresolved."""
import sys


def main():
    sys.stdin.read()  # consume input, we don't need it
    print(0)


if __name__ == "__main__":
    main()
