# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    N = int(data[p]); T = int(data[p + 1]); cap = int(data[p + 2])
    p += 5  # skip N T cap w M header (w, M unused)
    trk = [min(T - 1, i // cap) for i in range(N)]
    sys.stdout.write(" ".join(str(x) for x in trk) + "\n")


if __name__ == "__main__":
    main()
