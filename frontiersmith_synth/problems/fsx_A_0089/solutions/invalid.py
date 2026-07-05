# TIER: invalid
# Emits points far outside the unit square -> fails feasibility -> scores 0.
import sys

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = []
    for i in range(n):
        out.append("%.10f %.10f" % (5.0 + i, -3.0 - i))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
