# TIER: invalid
# Declares (and writes) more policies than underwriting capacity C allows -- a hard
# feasibility violation that must score 0.
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0]); C = int(data[1])
    m = C + 3
    idxs = [i % N for i in range(m)]
    print(m)
    print(*idxs)


if __name__ == "__main__":
    main()
