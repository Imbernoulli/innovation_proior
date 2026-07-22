# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split("\n")
    R, A, k, p = [int(x) for x in data[0].split()]
    out = []
    for r in range(R):
        color = 1 + (r % 2)
        for a in range(A):
            out.append("%d %d" % (r, color))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
