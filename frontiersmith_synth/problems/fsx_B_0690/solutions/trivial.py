# TIER: trivial
# Reproduces the checker's own internal baseline construction EXACTLY:
# project 0 sits alone in bundle 1 (it is always safe), and every other
# project is lumped together into bundle 2.  No attempt is made to reason
# about which projects clash or which small projects could rescue a big one.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    P = int(next(it)); V = int(next(it)); K = int(next(it))
    # we don't need the per-project data at all for this construction
    assign = [2] * P
    if P > 0:
        assign[0] = 1
    print(P)
    print(" ".join(map(str, assign)))


if __name__ == "__main__":
    main()
