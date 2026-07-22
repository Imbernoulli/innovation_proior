#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the scenario-consensus battery commitment problem.

Feasibility (committed plan, all bids execute):
    |q[t]| <= Pmax;  E[0] = E0;
    E[t] = E[t-1] - q[t]           if q[t] > 0   (discharge delivers q[t])
    E[t] = E[t-1] + eta*(-q[t])    if q[t] < 0   (charge stores eta*|q[t]|)
    0 <= E[t] <= Emax  for all t.

Scenario profit (outage hours void the committed action; residual storage is
the option that keeps its terminal value):
    for t:  if o_s[t] == 1:  profit -= lam*|q[t]|
            elif q[t] > 0:   d = min(q[t], E);
                             profit += p_s[t]*d - mu*(q[t]-d);  E -= d
            elif q[t] < 0:   a = min(-q[t], (Emax-E)/eta);
                             profit -= p_s[t]*a + mu*((-q[t])-a);  E += eta*a
    profit += rho_s * E_end
F = min over scenarios.

Score:  B = F of the do-nothing plan (min_s rho_s*E0)  (>0);
        Ratio = clamp( F / (10*B), 0, 1 )   == min(1000, 100*F/B)/1000.
"""
import sys
import math

MAX_BYTES = 4 * 1024 * 1024
TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(inf):
    try:
        with open(inf, "rb") as fh:
            raw = fh.read(MAX_BYTES + 1)
    except Exception:
        fail("cannot read instance")
    if len(raw) > MAX_BYTES:
        fail("instance too large")
    try:
        lines = raw.decode("utf-8").split("\n")
        idx = 0

        def next_line():
            nonlocal idx
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if idx >= len(lines):
                raise ValueError("unexpected eof")
            ln = lines[idx]
            idx += 1
            return ln.split()

        tok = next_line()
        T, S = int(tok[0]), int(tok[1])
        if not (1 <= T <= 2000 and 1 <= S <= 200):
            raise ValueError("bad T/S")
        tok = next_line()
        Emax, Pmax, eta, E0, lam, mu = (float(v) for v in tok[:6])
        if not (Emax > 0 and Pmax > 0 and 0 < eta <= 1.0 and 0 <= E0 <= Emax
                and lam >= 0 and mu >= 0):
            raise ValueError("bad params")
        prices, outages, rho = [], [], []
        for _ in range(S):
            rho.append(float(next_line()[0]))
            p = [float(v) for v in next_line()]
            o = [int(v) for v in next_line()]
            if len(p) != T or len(o) != T or any(v not in (0, 1) for v in o):
                raise ValueError("bad scenario block")
            if any((v != v or v in (float("inf"), float("-inf"))) for v in p):
                raise ValueError("non-finite price")
            prices.append(p)
            outages.append(o)
    except Exception:
        fail("malformed instance")
    return T, S, Emax, Pmax, eta, E0, lam, mu, prices, outages, rho


def simulate(T, S, Emax, Pmax, eta, E0, lam, mu, prices, outages, rho, q):
    """Return (worst_scenario_profit, committed_feasible)."""
    E = E0
    for t in range(T):
        v = q[t]
        if v > 0:
            E -= v
        elif v < 0:
            E += eta * (-v)
        if E < -TOL or E > Emax + TOL:
            return None, False
    worst = None
    for s in range(S):
        p, o = prices[s], outages[s]
        E = E0
        prof = 0.0
        for t in range(T):
            v = q[t]
            if o[t]:
                prof -= lam * (v if v >= 0 else -v)
            elif v > 0:
                d = v if v <= E else E
                if d < 0:
                    d = 0.0
                prof += p[t] * d - mu * (v - d)
                E -= d
            elif v < 0:
                head = (Emax - E) / eta
                if head < 0:
                    head = 0.0
                a = -v if -v <= head else head
                prof -= p[t] * a + mu * ((-v) - a)
                E += eta * a
        prof += rho[s] * E
        if worst is None or prof < worst:
            worst = prof
    return worst, True


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]
    T, S, Emax, Pmax, eta, E0, lam, mu, prices, outages, rho = read_instance(inf)

    # ---- participant artifact: exactly T finite floats ----
    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_BYTES:
        fail("output too large")
    toks = raw.decode("utf-8", "replace").split()
    if len(toks) != T:
        fail("expected %d values, got %d" % (T, len(toks)))
    q = []
    for tok in toks:
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token")
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite value")
        if v > Pmax + TOL or v < -Pmax - TOL:
            fail("power limit violated")
        q.append(max(-Pmax, min(Pmax, v)))

    F, feas = simulate(T, S, Emax, Pmax, eta, E0, lam, mu,
                       prices, outages, rho, q)
    if not feas:
        fail("committed state-of-charge trajectory infeasible")

    B = min(rho[s] * E0 for s in range(S))
    if B <= 1e-9:
        fail("degenerate instance baseline")
    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    if ratio < 0.0:
        ratio = 0.0
    print("worst_scenario=%.6f baseline=%.6f  Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
