# TIER: invalid
# Emits an INFEASIBLE schedule: every epoch bolts p distinct FRESH lamps, so each
# transition adds p > Bmax new bolts -> violates reconfiguration bandwidth -> 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[1]); E = int(toks[2]); p = int(toks[4])
    out = []
    for t in range(E):
        keys = [(t * p + j) % N for j in range(p)]   # all-new set each epoch
        out.append(str(p) + " " + " ".join(str(k) for k in keys))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
