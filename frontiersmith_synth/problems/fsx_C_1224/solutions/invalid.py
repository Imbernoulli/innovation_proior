# TIER: invalid
"""Assigns every field the SAME tag (0) -- a maximal compatibility break:
no parser could ever tell these fields apart on the wire. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    V, M, t1cap, t2cap, t2cost, t3cost = (int(next(it)) for _ in range(6))
    out = []
    for fid in range(M):
        next(it); next(it); next(it); next(it)
        out.append(f"{fid} 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
