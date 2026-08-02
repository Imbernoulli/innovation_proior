#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (ans ignored) -- deterministic scorer.

Instance (<in>):
    line1: PRF K C
    line2: K aspect angles (degrees)
    line3: K observed (already PRF-folded) spectral-line frequencies (Hz)
    next C lines: name blade_count rate_min rate_max   (candidate target classes, RPS)

Submission (<out>): two whitespace-separated tokens
    class_id  rate
class_id must be a valid index into the C candidate rows (0-based, in the order
printed in THIS instance); rate must be finite and lie within that class's own
[rate_min, rate_max] (a physical-plausibility feasibility gate) -- otherwise
Ratio: 0.0.

Objective (maximize): re-fold the participant's forward-model prediction
    f_pred(theta_i) = blade_count[class_id] * rate * sin(theta_i)
at EVERY given aspect angle i and compare it (after applying the SAME PRF fold
used to build the observations) against the observed value. Each angle
contributes a score in [FLOOR, 1] that decays as the Hz mismatch grows past a
tolerance band, so an answer that only explains a single angle (the classic
"trust the raw peak, ignore aliasing" mistake) scores far worse than one that
is consistent across the whole aspect-angle spread. F = sum over angles.

B = the same scoring function applied to the checker's own naive one-shot
reference: candidate row 0 (as printed) at its own range midpoint. Final
score: ratio = min(1, F / (10*B)).
"""
import sys, math

FLOOR = 0.12   # per-angle score floor (keeps B off zero / keeps ratio well-defined)
CAP = 6.0      # Hz slack: score decays linearly to FLOOR as |mismatch| -> CAP


def out_ratio(v, reason=""):
    if reason:
        sys.stdout.write("# %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % v)
    sys.exit(0)


def fold(f, prf):
    k = math.floor(f / prf + 0.5)
    return abs(f - k * prf)


def angle_score(resid):
    return max(FLOOR, min(1.0, 1.0 - resid / CAP))


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    prf = float(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    angles = [float(toks[idx + i]) for i in range(K)]; idx += K
    obs = [float(toks[idx + i]) for i in range(K)]; idx += K
    classes = []
    for _ in range(C):
        name = toks[idx]; idx += 1
        blade = int(toks[idx]); idx += 1
        rlo = float(toks[idx]); idx += 1
        rhi = float(toks[idx]); idx += 1
        classes.append((name, blade, rlo, rhi))
    return prf, angles, obs, classes


def fit_quality(prf, angles, obs, blade, rate):
    F = 0.0
    for th, o in zip(angles, obs):
        s = math.sin(math.radians(th))
        f_pred = blade * rate * s
        f_pred_folded = fold(f_pred, prf)
        resid = abs(f_pred_folded - o)
        F += angle_score(resid)
    return F


def main():
    if len(sys.argv) < 3:
        out_ratio(0.0, "bad args")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        prf, angles, obs, classes = read_instance(inf)
    except Exception as e:
        out_ratio(0.0, "bad instance: %r" % (e,))

    K = len(angles)
    C = len(classes)
    if K <= 0 or C <= 0:
        out_ratio(0.0, "empty instance")

    # ---- internal baseline B: naive reference construction (row 0, its own midpoint) ----
    b0_name, b0_blade, b0_lo, b0_hi = classes[0]
    B = fit_quality(prf, angles, obs, b0_blade, (b0_lo + b0_hi) / 2.0)
    B = max(B, 1e-9)

    # ---- read participant output (bounded) ----
    try:
        with open(outf, "rb") as f:
            raw = f.read(65536)
    except Exception:
        out_ratio(0.0, "no output")
    text = raw.decode("utf-8", "replace")
    toks = text.split()
    if len(toks) != 2:
        out_ratio(0.0, "expected exactly 2 tokens (class_id rate), got %d" % len(toks))

    try:
        class_id = int(toks[0])
    except Exception:
        out_ratio(0.0, "class_id not an integer: %r" % toks[0])
    if not (0 <= class_id < C):
        out_ratio(0.0, "class_id %d out of range [0,%d)" % (class_id, C))

    try:
        rate = float(toks[1])
    except Exception:
        out_ratio(0.0, "rate not a float: %r" % toks[1])
    if not math.isfinite(rate):
        out_ratio(0.0, "rate not finite")

    name, blade, rlo, rhi = classes[class_id]
    eps = 1e-6 * max(1.0, abs(rhi))
    if not (rlo - eps <= rate <= rhi + eps):
        out_ratio(0.0, "rate %.6f out of class %r range [%.6f,%.6f]" % (rate, name, rlo, rhi))

    F = fit_quality(prf, angles, obs, blade, rate)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    sys.stdout.write("F=%.6f B=%.6f class=%s rate=%.6f Ratio: %.6f\n" % (F, B, name, rate, ratio))


if __name__ == "__main__":
    main()
