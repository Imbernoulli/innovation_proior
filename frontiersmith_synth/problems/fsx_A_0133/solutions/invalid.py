# TIER: invalid
# Emits infeasible output: N huge overlapping bubbles that leave the airspace and cover the
# no-fly zones -> checker must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); cx = float(toks[1]); cy = float(toks[2]); R = float(toks[3])
    out = [str(N)]
    for _ in range(N):
        out.append("%r %r %r" % (cx, cy, 3.0 * R))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
