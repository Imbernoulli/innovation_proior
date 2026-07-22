# TIER: strong
# INSIGHT: the generalizing hypothesis class is a RECURSION over a latent
# wear state, not a function of position/time. Back out a per-row wear
# ESTIMATE What_i from the given observation formula, then fit a genuine
# one-step recursion What_i ~= f(What_{i-1}, gap_i, load_i, material_i) by
# TEACHER FORCING (regress each step against the PREVIOUS estimate, not a
# self-rolled trajectory -- far more robust to per-row noise than trying to
# roll a candidate forward during fitting).
#
# SYMBOLIC MODEL SELECTION: instead of committing to one fixed shape, search
# a small library of candidate recursion FAMILIES --
#   (a) linear accumulation, no idle recovery, no saturation
#   (b) exponential idle-recovery decay + linear load accumulation
#   (c) exponential idle-recovery decay + SATURATING load accumulation
#       (gain shrinks as the decayed previous wear approaches the cap)
# -- each with a small grid of numeric constants (decay rate, per-material
# gain), and keep whichever family+constants minimizes the in-sample one-step
# residual. This is what actually recovers the mechanism: idle gaps really do
# cause recovery and heavy load really does saturate, so family (c) wins on
# real training logs, and unlike the greedy time-trend it is expressed
# entirely in terms of (Wprev, gap, load, material) so it transfers to any
# held-out horizon or load/gap mix.
import sys
import math


def read_input():
    data = sys.stdin.read().split()
    pos = 0
    t = int(data[pos]); pos += 1
    n = int(data[pos]); pos += 1
    alpha = float(data[pos]); pos += 1
    base = [float(data[pos]), float(data[pos + 1]), float(data[pos + 2])]; pos += 3
    rows = []
    for i in range(n):
        gap = float(data[pos]); load = float(data[pos + 1])
        mat = int(data[pos + 2]); T = float(data[pos + 3])
        pos += 4
        rows.append((gap, load, mat, T))
    return t, n, alpha, base, rows


A_GRID = [0.0] + [0.02 + 0.03 * k for k in range(17)]  # 0, 0.02 .. 0.50


def fit_candidate(rows, whats, use_decay, use_saturation):
    """One-step teacher-forced fit: What_i ~= Wprev*exp(-a*gap_i)
    + b_mat[m] * load_i^1.5 * (optionally) (1 - decayed_prev).
    Grid-search a (0 if use_decay is False), closed-form least-squares b per
    material given a. Returns (sse, a, b_mat)."""
    n = len(rows)
    best = None
    a_values = A_GRID if use_decay else [0.0]
    for a in a_values:
        num = [0.0, 0.0, 0.0]
        den = [0.0, 0.0, 0.0]
        for i in range(1, n):
            gap, load, mat, _T = rows[i]
            wprev = whats[i - 1]
            decayed = wprev * math.exp(-a * gap)
            x = (load ** 1.5) * ((1.0 - decayed) if use_saturation else 1.0)
            r = whats[i] - decayed
            num[mat] += x * r
            den[mat] += x * x
        b_mat = [ (num[m] / den[m]) if den[m] > 1e-9 else 0.0 for m in range(3) ]
        sse = 0.0
        for i in range(1, n):
            gap, load, mat, _T = rows[i]
            wprev = whats[i - 1]
            decayed = wprev * math.exp(-a * gap)
            x = (load ** 1.5) * ((1.0 - decayed) if use_saturation else 1.0)
            pred = decayed + b_mat[mat] * x
            r = whats[i] - pred
            sse += r * r
        if best is None or sse < best[0]:
            best = (sse, a, b_mat)
    return best


def build_expr(a, b_mat, use_decay, use_saturation):
    if use_decay:
        decay_term = "Wprev*exp(-%.8f*gap)" % a
    else:
        decay_term = "Wprev"
    mat_coef = "(%.8f*m0+%.8f*m1+%.8f*m2)" % (b_mat[0], b_mat[1], b_mat[2])
    if use_saturation:
        gain_term = "%s*load**1.5*(1-%s)" % (mat_coef, decay_term)
    else:
        gain_term = "%s*load**1.5" % mat_coef
    return "%s + %s" % (decay_term, gain_term)


def main():
    t, n, alpha, base, rows = read_input()
    whats = [(T / base[mat] - 1.0) / alpha for (gap, load, mat, T) in rows]

    candidates = [
        (False, False),  # (a) linear accumulation, no recovery, no saturation
        (True, False),   # (b) idle-recovery decay + linear accumulation
        (True, True),    # (c) idle-recovery decay + saturating accumulation
    ]
    best_overall = None
    for use_decay, use_saturation in candidates:
        sse, a, b_mat = fit_candidate(rows, whats, use_decay, use_saturation)
        if best_overall is None or sse < best_overall[0]:
            best_overall = (sse, a, b_mat, use_decay, use_saturation)

    _sse, a, b_mat, use_decay, use_saturation = best_overall
    print(build_expr(a, b_mat, use_decay, use_saturation))


if __name__ == "__main__":
    main()
