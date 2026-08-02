#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_instance


def main():
    test_id = int(sys.argv[1])
    inst = build_instance(test_id)
    N = inst["N"]
    parent = inst["parent"]
    measurements = inst["measurements"]

    lines = []
    lines.append(f"{test_id} {N} {len(measurements)}")
    lines.append(" ".join(str(parent[v]) for v in range(1, N)))
    for v, dv in measurements:
        lines.append(f"{v} {dv:.6f}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
