# TIER: invalid
# Emits a topology that blows the link budget (claims every pair is directly
# linked, a complete graph) -- infeasible, must score 0.
import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    out = [str(len(edges))]
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
