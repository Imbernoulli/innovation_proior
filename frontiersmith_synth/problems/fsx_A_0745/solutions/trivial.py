# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0])
    S = data[1]
    print(1)
    print(" ".join(S[i] for i in range(n)))


if __name__ == "__main__":
    main()
