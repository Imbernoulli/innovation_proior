# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    # We don't even need to parse anything: do nothing, place zero firebreaks.
    print(0)
    print()


if __name__ == "__main__":
    main()
