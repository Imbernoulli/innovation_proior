# TIER: trivial
import sys


def main():
    sys.stdin.read()  # instance is unused: always emit the no-pipelining construction
    print(1)
    print(0)


if __name__ == "__main__":
    main()
