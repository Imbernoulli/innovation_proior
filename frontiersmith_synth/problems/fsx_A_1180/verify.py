#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the crystal-structure-from-peaks task.

Reads the test id from <in>'s header and reconstructs the hidden orthorhombic
crystal (a,b,c,centering) with the EXACT same seeded formula as gen.py (the
hidden law lives only here and in gen.py -- never in the input/output).

Score = weighted combination of:
  (1) lattice-constant accuracy vs hidden truth (axis-permutation invariant),
  (2) Bragg self-consistency of the submitted indexing (does h,k,l @ a,b,c
      reproduce the observed peak it claims to explain?) -- this is EASY to
      score well on with a wrong structure, by design (a wrong indexing can
      fit peak positions fine),
  (3) correctness of the submitted indexing against the hidden true indexing
      (partial credit -- there can be more than one valid true candidate per
      observed line),
  (4) F1 of the FULL predicted diffraction pattern (submitted a,b,c,centering,
      forward-generated out to the held-out 2theta range) against the hidden
      true pattern -- this is dominated by whether the systematic-absence
      signature (and hence the centering) was identified correctly, and it is
      weighted heavily.
"""
import sys, math

LAMBDA = 1.5406
MERGE_TOL_DEG = 0.05
TOL_FIT_DEG = 1.5          # how forgiving the self-consistency credit (2) is
TOL_MATCH_DEG = 0.15        # peak-matching tolerance for extrapolation F1 (3)/(4)
TOL_CHECK_DEG = 0.1         # (unused directly here; consistent with strong.py's own logic)
A_MAX = 40.0                # sanity bound on submitted lattice constants
HKL_MAX = 2000               # sanity bound on submitted indices


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def allowed(h, k, l, cent):
    if h == 0 and k == 0 and l == 0:
        return False
    if cent == "P":
        return True
    if cent == "I":
        return (h + k + l) % 2 == 0
    if cent == "F":
        return (h % 2 == k % 2) and (k % 2 == l % 2)
    return False


def true_structure(t):
    import random
    rng = random.Random(90000 + 97 * t)
    a, b, c = (float(v) for v in rng.sample(range(4, 14), 3))
    centering_seq = {1: "P", 2: "P", 3: "I", 4: "F", 5: "I",
                      6: "F", 7: "I", 8: "F", 9: "I", 10: "F"}
    cent = centering_seq.get(t, "P")
    theta2_given_max = 33.0 + 2.0 * (t - 1)
    theta2_full_max = theta2_given_max + 22.0
    return a, b, c, cent, theta2_given_max, theta2_full_max


def gen_spectrum(a, b, c, cent, theta2_cutoff, lam=LAMBDA):
    q_max = (2.0 * math.sin(math.radians(theta2_cutoff / 2.0)) / lam) ** 2
    raw = []
    hb = int(a * math.sqrt(q_max)) + 2
    for h in range(0, hb + 1):
        qh = (h / a) ** 2
        if qh > q_max + 1e-12:
            break
        kb = int(b * math.sqrt(max(0.0, q_max - qh))) + 2
        for k in range(0, kb + 1):
            qhk = qh + (k / b) ** 2
            if qhk > q_max + 1e-12:
                break
            lb = int(c * math.sqrt(max(0.0, q_max - qhk))) + 2
            for l in range(0, lb + 1):
                q = qhk + (l / c) ** 2
                if q > q_max + 1e-12:
                    break
                if h == 0 and k == 0 and l == 0:
                    continue
                if not allowed(h, k, l, cent):
                    continue
                sin_t = lam * math.sqrt(q) / 2.0
                if sin_t > 1.0:
                    continue
                theta2 = 2.0 * math.degrees(math.asin(sin_t))
                raw.append((theta2, (h, k, l)))
    raw.sort(key=lambda x: x[0])
    groups = []
    for theta2, hkl in raw:
        if groups and theta2 - groups[-1][0] <= MERGE_TOL_DEG:
            groups[-1][1].append(hkl)
        else:
            groups.append([theta2, [hkl]])
    return groups


def theta2_of(h, k, l, a, b, c, lam=LAMBDA):
    q = (h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2
    sin_t = lam * math.sqrt(q) / 2.0
    if sin_t > 1.0 + 1e-9:
        return None
    sin_t = min(1.0, sin_t)
    return 2.0 * math.degrees(math.asin(sin_t))


def main():
    inp_tokens = open(sys.argv[1]).read().split()
    try:
        it = iter(inp_tokens)
        t = int(next(it))
        lam = float(next(it))
        theta2_given_max = float(next(it))
        theta2_full_max = float(next(it))
        M = int(next(it))
        given_theta2 = [float(next(it)) for _ in range(M)]
    except Exception:
        fail("bad input")

    a_true, b_true, c_true, cent_true, gmax_chk, fmax_chk = true_structure(t)
    # sanity: our own generator/checker must agree on the window (defensive, not participant-facing)
    if abs(gmax_chk - theta2_given_max) > 1e-6 or abs(fmax_chk - theta2_full_max) > 1e-6:
        fail("internal window mismatch")

    true_groups_full = gen_spectrum(a_true, b_true, c_true, cent_true, theta2_full_max, lam)
    true_given_groups = [g for g in true_groups_full if g[0] <= theta2_given_max + 1e-9]
    if len(true_given_groups) != M:
        fail("internal generation mismatch")
    true_full_positions = [g[0] for g in true_groups_full]

    # ---- parse participant output ----
    out_tokens = open(sys.argv[2]).read().split()
    try:
        ot = iter(out_tokens)
        a_s = float(next(ot))
        b_s = float(next(ot))
        c_s = float(next(ot))
        cent_s = next(ot)
        hkl_sub = []
        for _ in range(M):
            h = next(ot); k = next(ot); l = next(ot)
            hkl_sub.append((h, k, l))
    except StopIteration:
        fail("too few tokens")
    except Exception:
        fail("parse error")

    for v in (a_s, b_s, c_s):
        if not math.isfinite(v):
            fail("non-finite lattice constant")
    if not (0.0 < a_s <= A_MAX and 0.0 < b_s <= A_MAX and 0.0 < c_s <= A_MAX):
        fail("lattice constant out of range")
    if cent_s not in ("P", "I", "F"):
        fail("bad centering token %r" % cent_s)

    hkl_int = []
    for (h, k, l) in hkl_sub:
        try:
            hi_, ki_, li_ = int(h), int(k), int(l)
        except Exception:
            fail("non-integer h,k,l")
        if hi_ < 0 or ki_ < 0 or li_ < 0 or hi_ > HKL_MAX or ki_ > HKL_MAX or li_ > HKL_MAX:
            fail("h,k,l out of range")
        if hi_ == 0 and ki_ == 0 and li_ == 0:
            fail("h=k=l=0")
        hkl_int.append((hi_, ki_, li_))

    # declared centering must be self-consistent with the submitted indexing
    for (hi_, ki_, li_) in hkl_int:
        if not allowed(hi_, ki_, li_, cent_s):
            fail("h,k,l=%d,%d,%d violates declared centering %s" % (hi_, ki_, li_, cent_s))

    # ---------- (1) lattice-constant accuracy, axis-permutation invariant ----------
    # Orthorhombic axis labeling (which length is called a/b/c) is a free convention, so
    # find the axis relabeling (index permutation sigma) that best aligns the submitted
    # a,b,c with the hidden truth; the SAME sigma is then used to relabel submitted
    # (h,k,l) triples before checking indexing correctness against the hidden truth (3).
    import itertools
    true_abc = (a_true, b_true, c_true)
    sub_abc = (a_s, b_s, c_s)
    best_err = None
    best_sigma = (0, 1, 2)
    for sigma in itertools.permutations(range(3)):
        err = sum(abs(sub_abc[sigma[i]] - true_abc[i]) / true_abc[i] for i in range(3)) / 3.0
        if best_err is None or err < best_err:
            best_err = err
            best_sigma = sigma
    lattice_score = math.exp(-best_err / 0.02)

    # ---------- (2) Bragg self-consistency of submitted indexing ----------
    fit_terms = []
    for i, (hi_, ki_, li_) in enumerate(hkl_int):
        th_pred = theta2_of(hi_, ki_, li_, a_s, b_s, c_s, lam)
        if th_pred is None:
            fit_terms.append(0.0)
            continue
        diff = abs(th_pred - given_theta2[i])
        fit_terms.append(math.exp(-diff / TOL_FIT_DEG))
    fit_score = sum(fit_terms) / M if M else 0.0

    # ---------- (3) indexing correctness against hidden truth ----------
    idx_hits = 0
    for i, (hi_, ki_, li_) in enumerate(hkl_int):
        hkl_vec = (hi_, ki_, li_)
        relabeled = tuple(hkl_vec[best_sigma[j]] for j in range(3))
        true_cands = set(true_given_groups[i][1])
        if relabeled in true_cands:
            idx_hits += 1
    idx_score = idx_hits / M if M else 0.0

    # ---------- (4) extrapolation F1 over the FULL (given+held-out) range ----------
    pred_groups = gen_spectrum(a_s, b_s, c_s, cent_s, theta2_full_max, lam)
    pred_positions = [g[0] for g in pred_groups]
    true_sorted = true_full_positions  # already ascending
    pred_sorted = pred_positions

    matched_pred = [False] * len(pred_sorted)
    TP = 0
    j0 = 0
    for tt in true_sorted:
        while j0 < len(pred_sorted) and pred_sorted[j0] < tt - TOL_MATCH_DEG:
            j0 += 1
        k_ = j0
        found = False
        while k_ < len(pred_sorted) and pred_sorted[k_] <= tt + TOL_MATCH_DEG:
            if not matched_pred[k_]:
                matched_pred[k_] = True
                found = True
                break
            k_ += 1
        if found:
            TP += 1
    denom = len(true_sorted) + len(pred_sorted)
    extrap_f1 = (2.0 * TP / denom) if denom > 0 else 1.0

    F = 0.10 * lattice_score + 0.10 * fit_score + 0.10 * idx_score + 0.70 * extrap_f1
    ratio = min(1.0, 0.90 * F)

    print("lattice=%.4f fit=%.4f idx=%.4f extrapF1=%.4f F=%.4f Ratio: %.6f" %
          (lattice_score, fit_score, idx_score, extrap_f1, F, ratio))


if __name__ == "__main__":
    main()
