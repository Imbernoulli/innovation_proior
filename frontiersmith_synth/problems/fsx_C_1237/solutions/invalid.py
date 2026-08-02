# TIER: invalid
"""Deliberately infeasible: starts at the RIGHTMOST relation and then asks
to extend right again, which immediately walks off the end of the chain.
Token schema is otherwise well-formed, so the checker must catch the
out-of-bounds move specifically, not just a malformed token count."""
import sys


def main():
    itok = sys.stdin.read().split()
    idx = 0
    tid = int(itok[idx]); idx += 1
    n = int(itok[idx]); idx += 1
    idx += n
    idx += 2
    h = int(itok[idx]); idx += 1

    pre = ['R'] * (h - 1)           # illegal from s=n-1: extends past relation n-1
    tail = ['R'] * (n - h)

    out = ["START", str(n - 1), "PRE"] + pre
    out += ["BRANCH", "LOW"] + tail
    out += ["BRANCH", "MID"] + tail
    out += ["BRANCH", "HIGH"] + tail
    print(" ".join(out))


if __name__ == "__main__":
    main()
