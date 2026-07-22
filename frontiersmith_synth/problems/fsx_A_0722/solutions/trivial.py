# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    N = int(header[0])
    print(" ".join(["0"] * N))


main()
