#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (<ans> is an unused placeholder)

Scores an amplifier-placement + launch-power plan for a straight optical
line against the "reach" (count of downstream nodes with SNR >= threshold).
Prints exactly one line ending in "Ratio: <float in [0,1]>".
"""
import sys
import math


def fail(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(toks):
    out = []
    for tk in toks:
        try:
            v = int(tk)
        except ValueError:
            return None
        out.append(v)
    return out


def read_floats(toks):
    out = []
    for tk in toks:
        try:
            v = float(tk)
        except ValueError:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out


def reach_count(N, xs, alpha, c0, c_ase, c_kerr, thresh, active_idx, powers):
    """Simulate the cascade and count downstream nodes (1..N-1) with SNR>=thresh.
    active_idx: sorted list of node indices carrying an amplifier (active_idx[0]==0).
    powers: launch power chosen for each active amplifier's outgoing span
            (len(powers)==len(active_idx))."""
    m = len(active_idx)
    reach = 0
    kerr_sum = 0.0   # sum of ell_prevspan * P_prevspan^3 over CLOSED prior spans
    amp_count = 0
    for k in range(m):
        s_idx = active_idx[k]
        e_idx = active_idx[k + 1] if k + 1 < m else N - 1
        P = powers[k]
        x_amp = xs[s_idx]
        amp_count += 1
        # noise present right at this amplifier's output (distance 0 into the span):
        # fixed per-amp noise-figure floor (one term per amplifier crossed so far)
        # + continuous ASE accumulated over distance traveled so far (x_amp)
        # + Kerr noise banked from all previously CLOSED spans.
        base = c0 * amp_count + c_ase * x_amp + c_kerr * kerr_sum
        lo = s_idx + 1
        hi = e_idx if k + 1 < m else N - 1
        for j in range(lo, hi + 1):
            d = xs[j] - x_amp
            noise = base + c_ase * d + c_kerr * d * (P ** 3)
            atten = 10.0 ** (-alpha * d / 10.0)
            sig = P * atten
            snr = sig / max(noise, 1e-9)
            if snr >= thresh:
                reach += 1
        if k + 1 < m:
            ell = xs[e_idx] - x_amp
            kerr_sum += ell * (P ** 3)
    return reach


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        in_toks = f.read().split()
    ptr = 0
    N = int(in_toks[ptr]); ptr += 1
    if N < 2:
        fail("corrupt input instance")
    xs = [int(v) for v in in_toks[ptr:ptr + N]]; ptr += N
    alpha = float(in_toks[ptr]); ptr += 1
    c0 = float(in_toks[ptr]); ptr += 1
    c_ase = float(in_toks[ptr]); ptr += 1
    c_kerr = float(in_toks[ptr]); ptr += 1
    thresh = float(in_toks[ptr]); ptr += 1
    K = int(in_toks[ptr]); ptr += 1
    allowed = [int(v) for v in in_toks[ptr:ptr + K]]; ptr += K
    if len(xs) != N or len(allowed) != K or K < 1:
        fail("corrupt input instance")
    if xs[0] != 0 or any(xs[i] >= xs[i + 1] for i in range(N - 1)):
        fail("corrupt input instance")
    allowed_set = set(allowed)

    # --- internal baseline B: amplify EVERY candidate node, all at the
    #     SMALLEST allowed power (dense-but-timid, a plain feasible plan). ---
    idxs_all = list(range(N))
    B = reach_count(N, xs, alpha, c0, c_ase, c_kerr, thresh, idxs_all, [allowed[0]] * N)
    B = max(B, 1)  # guard against a degenerate all-zero baseline

    # --- read participant output defensively ---------------------------
    try:
        with open(out_path, "r") as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")
    toks = raw.split()
    if len(toks) > 3 + 4 * N:
        fail("output too long")
    if len(toks) < 1:
        fail("empty output")

    m_tok = read_ints([toks[0]])
    if m_tok is None:
        fail("non-integer m")
    m = m_tok[0]
    if not (1 <= m <= N):
        fail("amplifier count out of range")

    need = 1 + 2 * m
    if len(toks) < need:
        fail("truncated output")

    idx_toks = read_ints(toks[1:1 + m])
    if idx_toks is None:
        fail("non-integer amplifier indices")
    pow_vals = read_floats(toks[1 + m:1 + 2 * m])
    if pow_vals is None:
        fail("non-finite or non-numeric power values")

    active_idx = idx_toks
    if active_idx[0] != 0:
        fail("first amplifier must be node 0 (the transmitter)")
    for i in range(m):
        if not (0 <= active_idx[i] <= N - 1):
            fail("amplifier index out of range")
    for i in range(1, m):
        if active_idx[i] <= active_idx[i - 1]:
            fail("amplifier indices must be strictly increasing")

    powers = []
    for pv in pow_vals:
        pi = int(round(pv))
        if abs(pv - pi) > 1e-6 or pi not in allowed_set:
            fail("power level not in the allowed discrete set")
        powers.append(pi)

    F = reach_count(N, xs, alpha, c0, c_ase, c_kerr, thresh, active_idx, powers)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%d B=%d m=%d" % (F, B, m))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
