# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    # S out of the legal [1,N] range -> the checker must reject this.
    S = N + 5
    print(S)
    print(" ".join(str(i) for i in range(1, S)))
    print(0)


if __name__ == "__main__":
    main()
