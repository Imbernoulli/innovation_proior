#!/usr/bin/env python3
"""Trivial baseline: emit the UNIFORM-GRID placement (the scorer's own reference).

Reads an instance from stdin, writes k integer facility coordinates using the
same near-square grid placement that score.py uses as its reference. By
construction this scores ~1_000_000. Used only in self-verification to confirm
the continuous-relax + snap solver strictly beats the trivial baseline.
"""
import sys
import score as S


def main():
    toks = sys.stdin.read().split()
    k = int(toks[0])
    L = int(toks[1])
    pts = S.grid_reference(k, L)
    out = [f"{x} {y}" for (x, y) in pts]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
