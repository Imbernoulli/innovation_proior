# TIER: greedy
# The obvious recipe: EXPECTED-VALUE threshold dispatch.  Average the
# published scenarios into a mean price curve, then run the classic
# charge-low / discharge-high quantile policy at full power with full depth.
# It never asks which scenario voids which hour -- that is the trap.
import sys


def read_instance():
    tok = sys.stdin.read().split()
    pos = 0
    T = int(tok[pos]); S = int(tok[pos + 1]); pos += 2
    Emax = float(tok[pos]); Pmax = float(tok[pos + 1]); eta = float(tok[pos + 2])
    E0 = float(tok[pos + 3]); lam = float(tok[pos + 4]); mu = float(tok[pos + 5])
    pos += 6
    prices, outages = [], []
    for _ in range(S):
        pos += 1  # rho (ignored by expected-value dispatch)
        p = [float(v) for v in tok[pos:pos + T]]; pos += T
        o = [int(v) for v in tok[pos:pos + T]]; pos += T
        prices.append(p)
        outages.append(o)
    return T, S, Emax, Pmax, eta, E0, prices


def main():
    T, S, Emax, Pmax, eta, E0, prices = read_instance()
    meanp = [sum(prices[s][t] for s in range(S)) / S for t in range(T)]
    sp = sorted(meanp)
    lo = sp[int(0.24 * (T - 1))]
    hi = sp[int(0.76 * (T - 1))]
    q = [0.0] * T
    E = E0
    floor = 0.10 * Emax          # naive depth-of-discharge guard
    for t in range(T):
        if meanp[t] >= hi and E > floor + 1e-9:
            v = min(Pmax, E - floor)
            q[t] = v
            E -= v
        elif meanp[t] <= lo:
            room = (Emax - E) / eta
            if room > 1e-9:
                v = -min(Pmax, room)
                q[t] = v
                E += eta * (-v)
    sys.stdout.write(" ".join(repr(v) for v in q) + "\n")


if __name__ == "__main__":
    main()
