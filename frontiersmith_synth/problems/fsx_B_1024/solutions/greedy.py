# TIER: greedy
# The obvious "average strong coder" approach: track the paired heat demand
# POINTWISE, period by period, and never touch the tank. Each period's heat
# demand is guaranteed satisfiable by a single operating point in the
# envelope slice, so Q_t = DQ[t], dump_t = 0 is always feasible -- but
# whenever DQ[t] sits strictly inside [L(t), U(t)] (the common case) this
# incurs the concave part-load fuel penalty every single period, instead of
# riding the (free) envelope vertices and letting the tank reconcile the
# mismatch over time.
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

    out = []
    for t in range(T):
        L, U = envelope(DP[t], Qcap, alpha, beta, Gamma, delta, Epsilon)
        Q = min(U, max(L, DQ[t]))   # clamp is a no-op: DQ[t] is guaranteed in [L,U]
        out.append("%.6f %.6f" % (Q, 0.0))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
