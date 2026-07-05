# TIER: invalid
# Emits probes far outside the belt map [0,1]^2 -> must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    M = int(toks[1]); K = int(toks[2])
    nf = M - K
    out = ["2.5 2.5" for _ in range(nf)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
