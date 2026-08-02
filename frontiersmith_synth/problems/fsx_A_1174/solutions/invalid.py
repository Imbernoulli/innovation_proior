# TIER: invalid
"""Infeasible artifact: emits a non-finite rate for every edge. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    ptr += 1  # testid
    n_species = int(data[ptr]); n_modules = int(data[ptr + 1]); n_edges = int(data[ptr + 2])
    ptr += 5
    ptr += 4 * n_modules
    edge_ids = []
    for _ in range(n_edges):
        eid = int(data[ptr]); ptr += 4
        edge_ids.append(eid)
    out = []
    for eid in edge_ids:
        out.append("%d nan" % eid)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
