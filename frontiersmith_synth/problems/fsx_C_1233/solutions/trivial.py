# TIER: trivial
"""Never cut: one giant block covering the whole corpus. This is exactly the
checker's own internal baseline construction, so this reproduces Ratio~0.1."""
import sys


def main():
    sys.stdin.read()  # instance is irrelevant to this construction
    print("RESULT 0 0 0 0 0 0 0 0 0")


if __name__ == "__main__":
    main()
