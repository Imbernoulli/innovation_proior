# TIER: invalid
"""Deliberately infeasible: claims to pick every part with BOTH arms (massive
duplicate assignment + guaranteed mutex/deadline violations) -- must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)

    def nx():
        return int(next(it))

    W = nx(); zlo = nx(); zhi = nx()
    posL0 = nx(); posR0 = nx()
    n = nx()
    # don't even bother reading the rest; just emit a bogus, over-claiming plan
    # (every part double-booked to both arms, all departing at time 0)
    idxs = list(range(n))
    flat = " ".join(f"{i} 0" for i in idxs)
    print(len(idxs))
    print(flat)
    print(len(idxs))
    print(flat)


if __name__ == "__main__":
    main()
