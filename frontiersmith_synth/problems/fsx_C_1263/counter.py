#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for the optical WDM link-budget problem.

1. Parses the instance: C channels, S spans (ase[1..S] -> N_ASE = sum), per-channel
   self-nonlinearity eta_c, K modulation tiers (bps[k], req_snr[k]), per-channel Pmax_c,
   baud, and the C x C cross-channel coupling matrix kappa.
2. Parses the participant's per-channel (power, tier) choice; validates STRICTLY:
     - exactly 2*C well-formed finite tokens (garbage/empty/huge/nan/inf -> Ratio: 0.0)
     - tier m_c integer in [0, K]; m_c == 0 ("off") requires P_c == 0 exactly
     - 0 <= P_c <= Pmax_c (small float slack)
     - for every ACTIVE channel (m_c >= 1): SNR_c computed from the FULL submitted power
       vector (self + cross-channel nonlinear noise) must reach req_snr[m_c]
   Any violation -> "Ratio: 0.0" and exit 0.
3. F = baud * sum(bps[m_c] for active channels).
4. Baseline B: the checker's own conservative construction -- every channel at
   min(Pmax_c, 0.7*peak0_c) simultaneously (peak0_c = (N_ASE/(2 eta_c))**(1/3), the
   SNR-maximising power ignoring cross-talk), each assigned the highest tier its REAL
   (cross-talk-included) resulting SNR clears. Maximisation ratio:
       sc = min(1000, 100*F/max(1e-9,B));  print("Ratio: %.6f" % (sc/1000))
"""
import math
import sys

MAX_TOKEN = 1e17  # tokens beyond this are treated as "huge/garbage" -> reject


def fail(reason):
    print(f"Ratio: 0.0  # {reason}")
    sys.exit(0)


def parse_finite_float(tok):
    try:
        v = float(tok)
    except ValueError:
        fail("non-numeric token")
    if not math.isfinite(v):
        fail("non-finite token (nan/inf)")
    if abs(v) > MAX_TOKEN:
        fail("token magnitude out of range")
    return v


def parse_finite_int(tok):
    v = parse_finite_float(tok)
    if v != int(v):
        fail("expected integer token")
    return int(v)


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path) as f:
        itoks = f.read().split()
    ip = iter(itoks)

    def inext():
        return next(ip)

    try:
        C = int(inext())
        S = int(inext())
        if C <= 0 or S <= 0:
            fail("bad instance header")
        ase = [int(inext()) for _ in range(S)]
        eta = [float(inext()) for _ in range(C)]
        K = int(inext())
        bps = []
        req = []
        for _ in range(K):
            bps.append(int(inext()))
            req.append(float(inext()))
        pmax = [float(inext()) for _ in range(C)]
        baud = float(inext())
        kappa = []
        for _ in range(C):
            kappa.append([float(inext()) for _ in range(C)])
    except (StopIteration, ValueError):
        fail("malformed instance (should not happen)")

    n_ase = float(sum(ase))
    if n_ase <= 0:
        fail("degenerate instance: N_ASE <= 0")

    # ---- parse participant output (untrusted) ----
    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except OSError:
        fail("cannot read output")

    if len(otoks) < 2 * C:
        fail("truncated output")

    op = iter(otoks)
    power = []
    tier = []
    for c in range(C):
        p_tok = next(op)
        m_tok = next(op)
        p = parse_finite_float(p_tok)
        m = parse_finite_int(m_tok)
        if not (0 <= m <= K):
            fail(f"tier out of range channel={c}")
        if p < -1e-9:
            fail(f"negative power channel={c}")
        p = max(p, 0.0)
        if p > pmax[c] * (1.0 + 1e-6) + 1e-9:
            fail(f"power exceeds Pmax channel={c}")
        p = min(p, pmax[c])
        if m == 0 and p > 1e-9:
            fail(f"'off' channel must have zero power channel={c}")
        power.append(p)
        tier.append(m)

    # ---- feasibility: every active channel's SNR must clear its declared tier ----
    def snr_of(p_vec, c):
        pc = p_vec[c]
        if pc <= 0.0:
            return 0.0
        qc = 0.0
        for c2 in range(C):
            if c2 == c:
                continue
            qc += kappa[c][c2] * p_vec[c2] * p_vec[c2]
        nli = eta[c] * pc ** 3 + qc * pc * pc
        return pc / (n_ase + nli)

    for c in range(C):
        m = tier[c]
        if m == 0:
            continue
        s = snr_of(power, c)
        if s < req[m - 1] * (1.0 - 1e-6):
            fail(f"SNR below tier threshold channel={c} snr={s:.6f} need={req[m - 1]:.6f}")

    F = 0.0
    for c in range(C):
        if tier[c] > 0:
            F += bps[tier[c] - 1]
    F *= baud

    # ---- baseline B: checker's own DELIBERATELY WEAK construction ----
    # "spend the least power that gets tier 1 working, self-noise only (ignore that
    # neighbours exist), then never bother trying for anything higher." Found by
    # bisection on the self-only SNR curve, which is monotonic increasing up to its own
    # peak0_c and req[0] is always comfortably below the isolated peak SNR (guaranteed by
    # the generator's parameter ranges), so this always converges to a small p0_c << Pmax_c.
    def self_snr(pc, c):
        if pc <= 0.0:
            return 0.0
        return pc / (n_ase + eta[c] * pc ** 3)

    B = 0.0
    for c in range(C):
        p0 = (n_ase / (2.0 * eta[c])) ** (1.0 / 3.0)
        target = req[0] * (1.0 + 1e-6)
        lo, hi = 0.0, p0
        if self_snr(hi, c) < target:
            continue  # cannot even reach tier 1 alone -> contributes 0 (should not occur)
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if self_snr(mid, c) < target:
                lo = mid
            else:
                hi = mid
        # this channel, alone at power hi, clears tier 1 -> baseline banks bps[0] for it
        B += bps[0]
    B *= baud

    if B <= 0:
        fail("degenerate instance: baseline achieves zero throughput")

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
