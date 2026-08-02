# TIER: trivial
# Build just the minimal ring: connect i to (i+1) mod N. Always feasible
# (cost = N <= L_max by construction) and always connects every node. This is
# exactly the checker's own internal baseline construction.
import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    edges = [(i, (i + 1) % N) for i in range(N)]
    out = [str(len(edges))]
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
