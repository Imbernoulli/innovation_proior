# TIER: trivial
# Reproduces the checker baseline: all depots on the horizontal midline.
# Min pairwise distance = 1/n = B  ->  Ratio = 0.1
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    out = []
    for i in range(n):
        x = (i + 0.5) / n
        out.append("%.10f %.10f" % (x, 0.5))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
