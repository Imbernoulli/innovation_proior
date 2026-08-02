# TIER: trivial
# Reproduce the checker's own baseline: every multiplier at full precision
# (t=0, no compensation). Always feasible (zero error) but pays maximum area.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    K = int(next(it)); G = int(next(it)); S = int(next(it))
    TMAX = int(next(it)); COMP_EXTRA = int(next(it))
    for _ in range(TMAX + 1):
        next(it)
    for _ in range(G):
        next(it); next(it)
    # positions' sample data is irrelevant to the trivial config
    out = []
    for _ in range(K):
        out.append("0 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
