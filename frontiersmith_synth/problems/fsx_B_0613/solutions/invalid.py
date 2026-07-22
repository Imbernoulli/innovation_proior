# TIER: invalid
# Emits a network that does NOT sort (a handful of arbitrary comparators). Must score 0.
import sys

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    # a few comparators that leave most consistent inputs unsorted
    comps = [(0, n - 1)] if n >= 2 else []
    out = [str(len(comps))]
    for (a, b) in comps:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
