# TIER: invalid
# Emits points far outside the unit cube -> fails feasibility -> scores 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = []
    for i in range(n):
        out.append("%.10f %.10f %.10f" % (7.0 + i, -4.0 - i, 9.0 + i))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
