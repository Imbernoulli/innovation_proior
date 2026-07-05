# TIER: trivial
# Diagonal completion -- place all m new cameras on the main diagonal. This is
# EXACTLY the checker's internal baseline, so it scores ~0.1.
import sys

def main():
    toks = sys.stdin.read().split()
    M = int(toks[0]); k = int(toks[1])
    m = M - k
    out = []
    for i in range(m):
        t = (i + 0.5) / m
        out.append("%.10f %.10f" % (t, t))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
