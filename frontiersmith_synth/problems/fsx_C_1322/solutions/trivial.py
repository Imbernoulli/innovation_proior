# TIER: trivial
"""Baseline construction: cool straight to T_min on step 1 and hold it there for
the rest of the batch (maximum possible instantaneous supersaturation, as fast as
physically expressible), charged with seed option 1. This exactly reproduces the
checker's own internal reference baseline, so it should score ~0.1 on every case.
It reaches the required yield easily (maximum driving force converts the most
solute) but detonates the nucleation term instantly, producing a slurry of mostly
fine powder -- a small mean crystal size."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    T_min = float(data[2])
    out = ["1"] + [str(T_min)] * N
    print(" ".join(out))


if __name__ == "__main__":
    main()
