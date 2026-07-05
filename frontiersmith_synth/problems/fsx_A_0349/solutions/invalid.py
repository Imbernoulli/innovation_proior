# TIER: invalid
# Emit out-of-range garbage coordinates -> checker rejects -> score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    d = int(toks[0]); M = int(toks[1]); K = int(toks[2])
    A = M - K
    out = []
    for _ in range(A):
        out.append("5.0 -3.0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
