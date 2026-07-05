# TIER: invalid
# Emits infeasible output: N huge overlapping disks that leave the room and cover the racks.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2])
    out = [str(N)]
    for _ in range(N):
        out.append("%r %r %r" % (W * 0.5, H * 0.5, max(W, H)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
