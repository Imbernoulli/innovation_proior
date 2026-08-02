# TIER: trivial
"""Reproduces the checker's own safe reference: never cut over (serve every
read from the untouched OLD store) and, for whatever backfill does happen,
always protect it with a version check. Always feasible, never wastes any
throughput on speed -- this is exactly the checker's baseline B, so it scores
Ratio ~= 0.1 by construction."""
import sys


def main():
    head = sys.stdin.readline().split()
    K, T, M = int(head[0]), int(head[1]), int(head[2])
    sys.stdin.readline()  # baseline values, unused
    out = [str(T)]
    out.append(" ".join(["1"] * M))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
