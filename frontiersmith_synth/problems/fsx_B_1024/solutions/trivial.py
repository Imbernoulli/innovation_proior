# TIER: trivial
# Always operate at the TOP envelope vertex U(t) (max heat extraction), and
# dump whatever overflows the tank. This is always feasible (the instance
# guarantees DQ[t] <= U(t) for every t, so the running balance never goes
# negative) but wasteful: it overproduces heat almost every period and pays
# dump fees for the surplus. This reproduces the checker's own baseline B.
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

    s = S_init
    out = []
    for t in range(T):
        L, U = envelope(DP[t], Qcap, alpha, beta, Gamma, delta, Epsilon)
        Q = U
        raw = s + Q - DQ[t]
        if raw < 0.0:
            raw = 0.0
        dump = max(0.0, raw - Cap)
        s = min(Cap, raw)
        out.append("%.6f %.6f" % (Q, dump))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
