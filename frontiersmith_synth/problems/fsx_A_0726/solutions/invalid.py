# TIER: invalid
# Deliberately infeasible: claims to carve more cells than the budget allows.
import sys


def main():
    toks = sys.stdin.read().split()
    R, C, B, r_out, c_out = (int(toks[i]) for i in range(5))
    k = B + 5
    out = [str(k)]
    r, c = 0, 0
    for i in range(k):
        out.append("%d %d" % (r, c))
        c += 1
        if c >= C:
            c = 0
            r = (r + 1) % R
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
