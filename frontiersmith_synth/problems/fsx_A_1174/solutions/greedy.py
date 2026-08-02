# TIER: greedy
"""Textbook two-point rate estimate, applied uniformly to every edge.

For each directed edge (src -> dst) treat the SOURCE species as if it were
decaying in isolation (ignore any coupling / conservation structure) and
estimate its rate from the log-slope between the LAST TWO snapshots:

    k_hat = -(ln c_src(t_last) - ln c_src(t_prev)) / (t_last - t_prev)

If that raw estimate is unreliable (non-finite or too small to trust), fall
back to a generic reaction-class default (0.2 for a chain edge, 50.0 for a
pair edge) -- a plausible textbook rule of thumb.

This works fine for the slow single-direction "chain" edges (genuine
exponential decay, so the two-point slope is close to the true rate). But
for a fast reversible pair that has already equilibrated before the FIRST
snapshot, consecutive snapshots are (up to noise) identical, so the
log-slope is driven to ~0 on BOTH directions and the fallback default fires
symmetrically (kf_hat = kr_hat = 50.0) -- discarding the equilibrium ratio
that is actually visible in the data, and getting the absolute scale wrong
by up to an order of magnitude. This will fail to re-equilibrate a held-out
initial condition that starts far from balance.
"""
import sys, math

EPS_CHAIN = 0.2
EPS_PAIR = 50.0
K_MAX = 400.0
MIN_TRUST = 0.05  # |k_hat| below this is treated as noise, not signal


def main():
    data = sys.stdin.read().split()
    ptr = 0
    testid = int(data[ptr]); ptr += 1
    n_species = int(data[ptr]); n_modules = int(data[ptr + 1]); n_edges = int(data[ptr + 2])
    n_snap = int(data[ptr + 3])
    ptr += 5
    mod_type = {}
    for _ in range(n_modules):
        mid = int(data[ptr]); mtype = int(data[ptr + 1])
        ptr += 4
        mod_type[mid] = mtype
    edges = []  # (edge_id, module_id, src, dst)
    for _ in range(n_edges):
        eid = int(data[ptr]); mid = int(data[ptr + 1]); src = int(data[ptr + 2]); dst = int(data[ptr + 3])
        ptr += 4
        edges.append((eid, mid, src, dst))
    times = [float(data[ptr + i]) for i in range(n_snap)]
    ptr += n_snap
    snaps = []
    for _ in range(n_snap):
        row = [float(data[ptr + i]) for i in range(n_species)]
        ptr += n_species
        snaps.append(row)

    t_prev, t_last = times[-2], times[-1]
    dt = max(1e-9, t_last - t_prev)

    out = []
    for eid, mid, src, dst in edges:
        fallback = EPS_CHAIN if mod_type[mid] == 0 else EPS_PAIR
        c_prev = max(1e-12, snaps[-2][src])
        c_last = max(1e-12, snaps[-1][src])
        k_hat = -(math.log(c_last) - math.log(c_prev)) / dt
        if not math.isfinite(k_hat) or k_hat < MIN_TRUST:
            k_hat = fallback
        k_hat = min(K_MAX, k_hat)
        out.append("%d %.6f" % (eid, k_hat))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
