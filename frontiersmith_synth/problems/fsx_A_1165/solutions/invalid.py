# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, C, TMAX = (int(x) for x in data[0].split())
    # Seal everything in a single page at the very last possible tick --
    # blows past almost every record's individual deadline. Infeasible.
    ids = " ".join(str(i) for i in range(1, N + 1))
    sys.stdout.write(f"{TMAX} {N} {ids}\n")


if __name__ == "__main__":
    main()
