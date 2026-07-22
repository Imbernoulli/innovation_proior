# TIER: strong
# Insight: at any single period the fuel cost is a CONCAVE function of Q
# (base linear cost plus a "part-load" bump that vanishes at both envelope
# endpoints and peaks in the middle). Minimizing a concave function over a
# convex set is always achieved at an EXTREME POINT, so the fuel-optimal
# choice at any period is one of the two envelope vertices L(t)/U(t), never
# an interior point -- the tank exists to make hopping between vertices
# feasible against the ACTUAL (interior) demand trace, not to track it.
#
# We solve a discretized dynamic program over the tank level: at each period
# choose among {L(t), U(t), clamp(DQ[t])} (a strict superset of the greedy
# choice, so strong can never do worse than greedy), and let dump absorb any
# tank overflow. To keep the DP state space finite we snap the post-dump
# tank level down to the nearest grid point after every transition, treating
# the (tiny, sub-grid-cell) remainder as extra real dump -- so every DP
# transition corresponds to an ACTUAL feasible (Q,dump) pair, no rounding
# tricks are needed at the end: the reconstructed path is exactly feasible.
import sys


def envelope(P, Qcap, alpha, beta, Gamma, delta, Epsilon):
    U = min(Qcap, (Gamma - alpha * P) / beta)
    if U < 0.0:
        U = 0.0
    L = delta * P - Epsilon
    if L < 0.0:
        L = 0.0
    if L > U:
        L = U
    return L, U


BUCKETS = 240


def main():
    toks = sys.stdin.read().split()
    p = 0
    T = int(toks[p]); p += 1
    Cap = float(toks[p]); p += 1
    S_init = float(toks[p]); p += 1
    Pmin = float(toks[p]); p += 1
    Pmax = float(toks[p]); p += 1
    Qcap = float(toks[p]); p += 1
    alpha = float(toks[p]); p += 1
    beta = float(toks[p]); p += 1
    Gamma = float(toks[p]); p += 1
    delta = float(toks[p]); p += 1
    Epsilon = float(toks[p]); p += 1
    b = float(toks[p]); p += 1
    kappa = float(toks[p]); p += 1
    dumpfee = float(toks[p]); p += 1
    DP = [0.0] * T
    DQ = [0.0] * T
    for t in range(T):
        DP[t] = float(toks[p]); p += 1
        DQ[t] = float(toks[p]); p += 1

    L_arr = [0.0] * T
    U_arr = [0.0] * T
    for t in range(T):
        L_arr[t], U_arr[t] = envelope(DP[t], Qcap, alpha, beta, Gamma, delta, Epsilon)

    step = Cap / BUCKETS if Cap > 0 else 1.0

    def snap(val):
        # floor to grid, clamp to [0,BUCKETS]; returns (bucket, dump_extra)
        if val <= 0.0:
            return 0, 0.0
        nb = int(val / step + 1e-9)
        if nb > BUCKETS:
            nb = BUCKETS
        snap_val = nb * step
        extra = val - snap_val
        if extra < 0.0:
            extra = 0.0
        return nb, extra

    INF = float("inf")
    # layer: dict bucket -> best cost so far
    # parent[t]: dict bucket -> (prev_bucket_or_None, Q, dump)
    parents = [None] * (T + 1)

    # ---- t = 1 (from the exact S_init, not yet on the grid) ----------------
    layer = {}
    parents[1] = {}
    Lt, Ut = L_arr[0], U_arr[0]
    cand = sorted(set([Lt, Ut, min(Ut, max(Lt, DQ[0]))]))
    for Q in cand:
        raw = S_init + Q - DQ[0]
        if raw < -1e-9:
            continue
        raw = max(0.0, raw)
        dump_forced = max(0.0, raw - Cap)
        capped = min(raw, Cap)
        nb, dump_extra = snap(capped)
        total_dump = dump_forced + dump_extra
        bump = kappa * (Q - Lt) * (Ut - Q)
        cost = b * Q + bump + dumpfee * total_dump
        if nb not in layer or cost < layer[nb] - 1e-12:
            layer[nb] = cost
            parents[1][nb] = (None, Q, total_dump)

    # ---- t = 2 .. T ----------------------------------------------------
    for t in range(2, T + 1):
        Lt, Ut = L_arr[t - 1], U_arr[t - 1]
        dqt = DQ[t - 1]
        cand = sorted(set([Lt, Ut, min(Ut, max(Lt, dqt))]))
        new_layer = {}
        parents[t] = {}
        for bucket, cost0 in layer.items():
            val = bucket * step
            for Q in cand:
                raw = val + Q - dqt
                if raw < -1e-9:
                    continue
                raw = max(0.0, raw)
                dump_forced = max(0.0, raw - Cap)
                capped = min(raw, Cap)
                nb, dump_extra = snap(capped)
                total_dump = dump_forced + dump_extra
                bump = kappa * (Q - Lt) * (Ut - Q)
                step_cost = b * Q + bump + dumpfee * total_dump
                newcost = cost0 + step_cost
                if nb not in new_layer or newcost < new_layer[nb] - 1e-12:
                    new_layer[nb] = newcost
                    parents[t][nb] = (bucket, Q, total_dump)
        if not new_layer:
            # should not happen (clamp candidate is always feasible); ultra
            # -defensive fallback keeps the program from crashing.
            new_layer = {0: cost0}
            parents[t][0] = (bucket, Lt, 0.0)
        layer = new_layer

    # ---- backtrack ------------------------------------------------------
    best_bucket = min(layer, key=lambda k: layer[k])
    Qs = [0.0] * T
    Dumps = [0.0] * T
    b_ = best_bucket
    for t in range(T, 0, -1):
        prev_b, Q, dump = parents[t][b_]
        Qs[t - 1] = Q
        Dumps[t - 1] = dump
        b_ = prev_b

    out = ["%.6f %.6f" % (Qs[t], Dumps[t]) for t in range(T)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
