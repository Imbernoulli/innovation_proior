# TIER: trivial
"""Canonical non-adaptive plan: start at relation 0, always extend right.
Uses the SAME continuation for every bucket -- exactly the checker's own
internal baseline construction, so this reproduces Ratio ~= 0.1."""
import sys


def main():
    itok = sys.stdin.read().split()
    idx = 0
    tid = int(itok[idx]); idx += 1
    n = int(itok[idx]); idx += 1
    idx += n           # C[]
    idx += 2            # mem_cap, spill_mult
    h = int(itok[idx]); idx += 1
    # est/bound not needed

    pre = ['R'] * (h - 1)
    tail = ['R'] * (n - h)

    out = ["START", "0", "PRE"] + pre
    out += ["BRANCH", "LOW"] + tail
    out += ["BRANCH", "MID"] + tail
    out += ["BRANCH", "HIGH"] + tail
    print(" ".join(out))


if __name__ == "__main__":
    main()
