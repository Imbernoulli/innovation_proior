# TIER: trivial
# Reproduces the checker baseline B = 2^(n-2): fix the last two intersections to
# phase 0 and use only phases {0,1} on the first n-2 intersections.
import sys, itertools


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = []
    for bits in itertools.product("01", repeat=n - 2):
        out.append("".join(bits) + "00")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
