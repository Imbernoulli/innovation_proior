#!/usr/bin/env python3
"""Trivial baseline: emit the STRIPE partition (the scorer's own reference).

Reads an instance from stdin, writes one district id per cell (row-major) using
the same K-horizontal-band partition that score.py uses as its reference. By
construction this scores ~1_000_000. Used only in self-verification to confirm
the SA solver strictly beats the trivial baseline.
"""
import sys
import score as S


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); K = int(next(it))
    assign = S.stripe_assignment(H, W, K)
    out = []
    for r in range(H):
        out.append(" ".join(str(assign[r * W + c]) for c in range(W)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
