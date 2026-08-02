# TIER: trivial
"""Baseline construction: literally the checker's own reference guess --
candidate row 0 (as printed in this instance) at its own range midpoint.
Ignores every observation. Calibrated to land at ratio ~0.1."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    prf = float(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    idx += K  # angles
    idx += K  # obs
    name = toks[idx]; idx += 1
    blade = int(toks[idx]); idx += 1
    rlo = float(toks[idx]); idx += 1
    rhi = float(toks[idx]); idx += 1
    rate = (rlo + rhi) / 2.0
    sys.stdout.write("0 %.6f\n" % rate)


if __name__ == "__main__":
    main()
