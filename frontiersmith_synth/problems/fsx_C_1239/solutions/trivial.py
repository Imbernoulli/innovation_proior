# TIER: trivial
"""Do nothing clever: route every message bit to checksum group 0 and leave
groups 1..K-1 completely unused. This is exactly one global parity bit over
the whole message -- it reproduces the checker's own internal baseline B, so
it scores ~0.1 by construction."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    M = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    # SEED, PBURST, and the burst-length table are unused by this tier.

    out = ["0"] * M
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
