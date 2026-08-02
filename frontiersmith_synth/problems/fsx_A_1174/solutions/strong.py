# TIER: strong
"""Exploit identifiability, not just fit harder.

For a single-direction "chain" edge (module type 0) the source species is a
clean exponential decay for ALL time, so its rate is fully determined: fit
ln(c_src(t)) = ln(c0) - k*t by ordinary least squares over EVERY snapshot
(robust to the per-reading noise, unlike a 2-point slope).

For a reversible "pair" module (module type 1, edges u->v and v->u) the two
species already sit at their mass-action equilibrium at the FIRST snapshot
(T1 = times[0]): the data show a ratio v/u that stays essentially constant
across all snapshots. That constant ratio IS the determined quantity:
    kf/kr = v_eq / u_eq   (from kf*u_eq = kr*v_eq at equilibrium)
The absolute scale kf+kr is NOT determined by this data -- any sum large
enough to have equilibrated before T1 is equally consistent with every
snapshot. So instead of fitting it (which is fitting noise), we pick a scale
that is safely large relative to the observed T1 -- large enough to still
equilibrate a held-out initial condition that starts far out of balance --
and split it according to the identified ratio.
"""
import sys, math

EPS = 1e-3
K_MAX = 400.0
SAFETY_C = 13.0  # (kf+kr) chosen as SAFETY_C / T1, capped at K_MAX


def main():
    data = sys.stdin.read().split()
    ptr = 0
    testid = int(data[ptr]); ptr += 1
    n_species = int(data[ptr]); n_modules = int(data[ptr + 1]); n_edges = int(data[ptr + 2])
    n_snap = int(data[ptr + 3])
    ptr += 5
    modules = []
    for _ in range(n_modules):
        mid = int(data[ptr]); mtype = int(data[ptr + 1]); u = int(data[ptr + 2]); v = int(data[ptr + 3])
        ptr += 4
        modules.append({'id': mid, 'type': mtype, 'u': u, 'v': v, 'edges': []})
    by_mid = {m['id']: m for m in modules}
    for _ in range(n_edges):
        eid = int(data[ptr]); mid = int(data[ptr + 1]); src = int(data[ptr + 2]); dst = int(data[ptr + 3])
        ptr += 4
        by_mid[mid]['edges'].append((eid, src, dst))
    times = [float(data[ptr + i]) for i in range(n_snap)]
    ptr += n_snap
    snaps = []
    for _ in range(n_snap):
        row = [float(data[ptr + i]) for i in range(n_species)]
        ptr += n_species
        snaps.append(row)

    T1 = times[0]
    S_hat = min(K_MAX, SAFETY_C / max(1e-9, T1))

    out = []
    for m in modules:
        if m['type'] == 0:
            eid, src, dst = m['edges'][0]
            ts = times
            ys = [math.log(max(1e-12, snaps[i][src])) for i in range(n_snap)]
            n = len(ts)
            tbar = sum(ts) / n
            ybar = sum(ys) / n
            num = sum((ts[i] - tbar) * (ys[i] - ybar) for i in range(n))
            den = sum((ts[i] - tbar) ** 2 for i in range(n))
            slope = num / den if den > 1e-12 else 0.0
            k_hat = -slope
            if not math.isfinite(k_hat) or k_hat < EPS:
                k_hat = EPS
            k_hat = min(K_MAX, k_hat)
            out.append("%d %.6f" % (eid, k_hat))
        else:
            (eid_fwd, u, v), (eid_rev, v2, u2) = m['edges']
            ratios = []
            for i in range(n_snap):
                cu = max(1e-9, snaps[i][u])
                cv = max(1e-9, snaps[i][v])
                ratios.append(cv / cu)
            ratios.sort()
            r_hat = ratios[len(ratios) // 2]  # median: kf/kr = v_eq/u_eq
            kf_hat = S_hat * r_hat / (1.0 + r_hat)
            kr_hat = S_hat / (1.0 + r_hat)
            kf_hat = min(K_MAX, max(EPS, kf_hat))
            kr_hat = min(K_MAX, max(EPS, kr_hat))
            out.append("%d %.6f" % (eid_fwd, kf_hat))
            out.append("%d %.6f" % (eid_rev, kr_hat))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
