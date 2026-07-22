# TIER: trivial
# Full insertion-sort comparator network on all n wires -- ignores the partial
# order entirely (this is exactly the checker's baseline; scores ~0.1).
import sys

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    comps = []
    for i in range(1, n):
        for j in range(i, 0, -1):
            comps.append((j - 1, j))
    out = [str(len(comps))]
    for (a, b) in comps:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
