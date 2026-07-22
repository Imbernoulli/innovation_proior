# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    N, t, V = int(data[0]), int(data[1]), int(data[2])
    # Claim far more marks than the budget allows -- must be rejected.
    print(V + 5)
    for i in range(3):
        print(N + 999, N + 999)


if __name__ == "__main__":
    main()
