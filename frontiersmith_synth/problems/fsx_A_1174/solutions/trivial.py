# TIER: trivial
"""Flat per-reaction-class guess: report a generic "slow" default rate for
every chain edge and a generic "fast, 1:1" default rate for every pair edge,
ignoring all snapshot data. This exactly reproduces the checker's own
internal baseline construction."""
import sys

BASELINE_CHAIN_RATE = 0.2
BASELINE_PAIR_RATE = 50.0


def main():
    data = sys.stdin.read().split()
    ptr = 0
    testid = int(data[ptr]); ptr += 1
    n_species, n_modules, n_edges, n_snap = (int(data[ptr + i]) for i in range(4))
    ptr += 5  # n_species n_modules n_edges n_snap K_MAX
    mod_type = {}
    for _ in range(n_modules):
        mid = int(data[ptr]); mtype = int(data[ptr + 1])
        ptr += 4  # module_id type u v
        mod_type[mid] = mtype
    out = []
    for _ in range(n_edges):
        eid = int(data[ptr]); mid = int(data[ptr + 1])
        ptr += 4  # edge_id module_id src dst
        rate = BASELINE_CHAIN_RATE if mod_type[mid] == 0 else BASELINE_PAIR_RATE
        out.append("%d %.6f" % (eid, rate))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
