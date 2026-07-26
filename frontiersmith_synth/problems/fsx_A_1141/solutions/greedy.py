# TIER: greedy
"""
Greedy / textbook approach: a single global multi-linear power-law fit.

V = A * d^p * T^q  is linear in log-log coordinates:
log(V) = log(A) + p*log(d) + q*log(T). Ordinary least squares of log(V)
against (log d, log T) over ALL training rows gives an excellent in-sample
fit -- inside the safe window both hidden channels are comparable in size,
so one power law explains almost everything. But a single power law has
only one pair of exponents: it cannot bend onto either channel's own
saturated asymptote once extrapolated to the far corners of the (d,T)
square, and on the two corners where the winning channel is
instance-dependent it has no way to track which channel actually wins.
"""
import sys
import math


def solve3(M, V):
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det3(M)
    if abs(D) < 1e-30:
        return None
    out = []
    for col in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][col] = V[r]
        out.append(det3(Mi) / D)
    return out


def fit_loglinear(rows):
    # log(V) = a + p*log(d) + q*log(T)
    S0 = len(rows)
    Sld = Slt = Sldld = Sltlt = Sldlt = 0.0
    Tv = Tld = Tlt = 0.0
    for (d, T, v) in rows:
        ld = math.log(d); lt = math.log(T); lv = math.log(v)
        Sld += ld; Slt += lt
        Sldld += ld * ld; Sltlt += lt * lt; Sldlt += ld * lt
        Tv += lv; Tld += ld * lv; Tlt += lt * lv
    M = [[S0, Sld, Slt], [Sld, Sldld, Sldlt], [Slt, Sldlt, Sltlt]]
    Vv = [Tv, Tld, Tlt]
    sol = solve3(M, Vv)
    if sol is None:
        return 0.0, 1.0, 1.0
    a, p, q = sol
    return a, p, q


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    rows = []
    for _ in range(n):
        d = float(data[idx]); idx += 1
        T = float(data[idx]); idx += 1
        v = float(data[idx]); idx += 1
        rows.append((d, T, v))

    a, p, q = fit_loglinear(rows)
    A = math.exp(a)
    print("%.10g * powv(d,%.10g) * powv(T,%.10g)" % (A, p, q))


if __name__ == "__main__":
    main()
