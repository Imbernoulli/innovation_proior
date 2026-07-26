# TIER: trivial
# Pure negative-gravitropism: ignore light and touch entirely, always drift
# "up" (away from gravity), and never branch. This is exactly the checker's
# own internal baseline construction, so it scores ~0.1 by construction.
import sys


def main():
    sys.stdin.read()  # instance unused by this policy
    print("1 0 0")
    print("0")


if __name__ == "__main__":
    main()
