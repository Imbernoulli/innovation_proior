#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the heat-exchanger-network
(HEN) synthesis problem.

Input instance: NH hot streams (THs>THt, cool from THs to THt), NC cold streams
(TCt>TCs, heat from TCs to TCt), a minimum approach temperature DTMIN, hot/cold
utility unit costs CH/CC, and an area-cost coefficient A.

Participant output:
    M
    i_1 j_1 Q_1
    ...
    i_M j_M Q_M
(1-indexed hot i, cold j, duty Q>=0; a stream may appear in several matches --
duplicate (i,j) pairs are summed.) Each hot stream's duty not covered by matches
is served by cold utility; each cold stream's duty not covered by matches is
served by hot utility -- these are NOT declared, they are the checker-computed
residuals, so the only way to reduce utility cost is to submit real matches.

Feasibility (ANY violation -> Ratio: 0.0):
  * output parses: M is an int in [0, 20*NH*NC + 20]; every (i,j,Q) parses to a
    valid 1-indexed pair and a finite, non-negative Q;
  * every match (i,j) with summed Q>0 satisfies BOTH ends of the minimum-approach
    rule using the streams' full supply/target temperatures (a stream may be
    split into parallel branches that each see the stream's own full range):
        THs_i - TCt_j >= DTMIN   and   THt_i - TCs_j >= DTMIN
  * no hot stream's total matched duty exceeds its available duty D_i =
    CPh_i*(THs_i-THt_i); no cold stream's total matched duty exceeds its need
    E_j = CPc_j*(TCt_j-TCs_j).

Objective (minimize):
  For every used match, LMTD_ij is the log-mean temperature difference of the
  two match-end driving forces above (arithmetic mean if they are ~equal).
  AreaCost = A * sum(Q_ij / LMTD_ij).  UtilCost = CC*sum(cold utility) +
  CH*sum(hot utility).  F = UtilCost + AreaCost.

Baseline B (checker's own trivial feasible network: zero matches, everything on
utility) = CC*sum(D_i) + CH*sum(E_j) > 0 always.

Score (minimization, normalized so echoing the baseline maps near 0.06 and a
network that beats it by >~16.7x saturates):
    sc = min(1000, 60 * B / max(1e-9, F));  Ratio = sc / 1000
"""
import math
import sys

TOL = 1e-6


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def lmtd(d1, d2):
    if abs(d1 - d2) < 1e-6:
        return d1
    return (d1 - d2) / math.log(d1 / d2)


def main():
    try:
        inf, outf = sys.argv[1], sys.argv[2]
        itok = read_tokens(inf)
        pos = 0

        def nxt():
            nonlocal pos
            v = itok[pos]
            pos += 1
            return v

        NH = int(nxt())
        NC = int(nxt())
        if NH <= 0 or NC <= 0 or NH > 1000 or NC > 1000:
            fail("degenerate instance sizes")
        dtmin = float(nxt())
        cH = float(nxt())
        cC = float(nxt())
        a = float(nxt())
        if not (dtmin > 0 and cH > 0 and cC > 0 and a >= 0):
            fail("degenerate instance costs")

        hots = []
        for _ in range(NH):
            ths, tht, cp = float(nxt()), float(nxt()), float(nxt())
            if not (ths > tht and cp > 0):
                fail("malformed hot stream in input")
            hots.append((ths, tht, cp))
        colds = []
        for _ in range(NC):
            tcs, tct, cp = float(nxt()), float(nxt()), float(nxt())
            if not (tct > tcs and cp > 0):
                fail("malformed cold stream in input")
            colds.append((tcs, tct, cp))
    except Exception as e:
        fail("bad input file (%r)" % (e,))

    D = [cp * (ths - tht) for (ths, tht, cp) in hots]
    E = [cp * (tct - tcs) for (tcs, tct, cp) in colds]

    # ---- parse participant output (adversarially) ----
    try:
        otok = read_tokens(outf)
        if not otok:
            fail("empty output")
        opos = 0

        def onxt():
            nonlocal opos
            if opos >= len(otok):
                raise ValueError("truncated output")
            v = otok[opos]
            opos += 1
            return v

        M = int(onxt())
        if M < 0 or M > 20 * NH * NC + 20:
            fail("M=%d out of allowed range" % M)
        agg = {}
        for _ in range(M):
            i = int(onxt())
            j = int(onxt())
            qtok = onxt()
            try:
                q = float(qtok)
            except ValueError:
                fail("non-numeric duty %r" % qtok)
            if not math.isfinite(q):
                fail("non-finite duty")
            if q < -TOL:
                fail("negative duty")
            q = max(q, 0.0)
            if i < 1 or i > NH or j < 1 or j > NC:
                fail("match index (%d,%d) out of range" % (i, j))
            key = (i - 1, j - 1)
            agg[key] = agg.get(key, 0.0) + q
    except SystemExit:
        raise
    except Exception as e:
        fail("bad output file (%r)" % (e,))

    # ---- feasibility ----
    remH = list(D)
    remC = list(E)
    area = 0.0
    for (i, j), q in agg.items():
        if q <= TOL:
            continue
        ths, tht, _ = hots[i]
        tcs, tct, _ = colds[j]
        d1 = ths - tct
        d2 = tht - tcs
        if d1 < dtmin - TOL or d2 < dtmin - TOL:
            fail("match (%d,%d) violates minimum approach temperature" % (i + 1, j + 1))
        L = lmtd(d1, d2)
        if not (L > 0 and math.isfinite(L)):
            fail("degenerate LMTD for match (%d,%d)" % (i + 1, j + 1))
        area += a * q / L
        remH[i] -= q
        remC[j] -= q

    for i in range(NH):
        if remH[i] < -TOL:
            fail("hot stream %d over-committed (matched more than its available duty)" % (i + 1))
    for j in range(NC):
        if remC[j] < -TOL:
            fail("cold stream %d over-committed (matched more than its required duty)" % (j + 1))

    CU = sum(max(0.0, x) for x in remH)
    HU = sum(max(0.0, x) for x in remC)
    util = cC * CU + cH * HU
    if not math.isfinite(util) or not math.isfinite(area):
        fail("non-finite objective")
    F = util + area

    B = cC * sum(D) + cH * sum(E)
    sc = min(1000.0, 60.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f util=%.6f area=%.6f HU=%.6f CU=%.6f  Ratio: %.6f"
          % (F, B, util, area, HU, CU, sc / 1000.0))


if __name__ == "__main__":
    main()
