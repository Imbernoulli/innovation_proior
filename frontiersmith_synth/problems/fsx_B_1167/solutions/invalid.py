# TIER: invalid
"""Deliberately infeasible: emit density values far outside [0, RHO_MAX] (3x RHO_MAX
everywhere), which also blows well past the mass budget. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    test_id = int(data[idx]); idx += 1
    nx = int(data[idx]); idx += 1
    nz = int(data[idx]); idx += 1
    rho_max = float(data[idx]); idx += 1

    big = rho_max * 3.0
    out = []
    for _ in range(nz):
        out.append(" ".join("%.6f" % big for _ in range(nx)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
