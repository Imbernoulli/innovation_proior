# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, M, K = (int(x) for x in data[0].split())
    reqs = [int(x) for x in data[2].split()] if K > 0 else []

    out = []
    for r in reqs:
        for i in range(1, r + 1):
            out.append("C %d" % i)
            if i > 1:
                out.append("E %d" % (i - 1))
        out.append("U %d" % r)
        out.append("E %d" % r)
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
