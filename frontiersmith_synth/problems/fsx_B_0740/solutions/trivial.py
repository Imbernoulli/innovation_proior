# TIER: trivial
# Reproduces the checker's own reference plan: everything on arm 0, strictly sequential
# in input (topological) order. Always feasible; makespan == sum(d_i) == the checker's B.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    d = [int(next(it)) for _ in range(N)]
    _s = [int(next(it)) for _ in range(N)]
    E = int(next(it))
    for _ in range(E):
        next(it); next(it)

    out = []
    t = 0
    for i in range(N):
        out.append("0 %d" % t)
        t += d[i]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
