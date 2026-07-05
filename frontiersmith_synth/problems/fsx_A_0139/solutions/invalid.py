# TIER: invalid
# Emits cameras far outside the unit square -> fails the feasibility gate -> 0.
import sys

def main():
    toks = sys.stdin.read().split()
    M = int(toks[0]); k = int(toks[1])
    m = M - k
    out = []
    for i in range(m):
        out.append("%.10f %.10f" % (7.0 + i, -4.0 - i))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
