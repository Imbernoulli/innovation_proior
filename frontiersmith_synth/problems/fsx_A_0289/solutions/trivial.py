# TIER: trivial
# Diagonal ride-line: reproduces the checker's internal baseline -> ratio ~0.1.
import sys

def main():
    inp = sys.stdin.read().split()
    d = int(inp[0]); M = int(inp[1])
    out = []
    for i in range(M):
        v = (i + 0.5) / M
        out.append(" ".join("%.10f" % v for _ in range(d)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
