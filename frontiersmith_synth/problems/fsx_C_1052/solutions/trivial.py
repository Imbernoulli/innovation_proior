# TIER: trivial
import sys


def main():
    a, p, L = map(int, sys.stdin.read().split()[:3])
    word = "".join(str(i % a) for i in range(L))
    sys.stdout.write(word + "\n")


if __name__ == "__main__":
    main()
