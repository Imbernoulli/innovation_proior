# TIER: invalid
# Walks head 0 off the left end of the rail -- an out-of-bounds move -- so the
# checker's feasibility gate must reject it and score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    L = int(next(it)); H = int(next(it)); D = int(next(it)); S = int(next(it))
    K = int(next(it)); N = int(next(it))
    starts = [int(next(it)) for _ in range(H)]

    steps = starts[0] + 3
    out = [str(H), str(steps)]
    out.extend(["M -1"] * steps)
    for h in range(1, H):
        out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
