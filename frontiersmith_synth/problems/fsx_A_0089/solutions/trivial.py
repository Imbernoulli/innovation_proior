# TIER: trivial
# Diagonal placement -- exactly the checker's internal baseline, so it scores ~0.1.
import sys

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = []
    for i in range(n):
        t = (i + 0.5) / n
        out.append("%.10f %.10f" % (t, t))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
