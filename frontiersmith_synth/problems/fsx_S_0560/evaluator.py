#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0560 -- "Stripes that hold under a stretching gradient"
(family: morphogen-french-flag-grn; format B, quality-metric).

THEME.  A growing 1-D tissue reads a monotone morphogen gradient and must paint
itself into three stripes (a French-flag pattern) at fixed proportions 1:2:1 --
the anterior quarter labelled 0, the central half labelled 1, the posterior quarter
labelled 2.  BUT every embryo is a different size and the source is a different
strength: from instance to instance the gradient's AMPLITUDE, its OFFSET, its
SLOPE/shape exponent, and the tissue LENGTH are all randomly rescaled (seeded).  A
low-frequency developmental "bump" and cell-level readout noise further corrupt the
field.  The tissue must proportion its stripes correctly under this unknown
rescaling.

WHAT THE MODEL SUBMITS.  A gene-regulatory READOUT NETWORK -- a small static rule
that maps each cell's morphogen environment to a stripe label.  The SAME network is
then applied by this evaluator to a family of freshly rescaled gradients the
candidate never saw, and scored on how many cells it labels correctly.

The network is (deliberately) expressive enough to encode very different readout
philosophies:
  * "absolute": threshold the raw local concentration;
  * "relative": threshold the min-max-normalised concentration (affine-invariant);
  * "rank":     threshold the cell's concentration RANK fraction within the field
                (invariant to ANY monotone rescaling -- amplitude, offset, slope).
plus an optional neighbour-averaging window `smooth` (GRN cells read neighbours to
denoise) and two cut points.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "L": int, "field": [float, ... length L]}
          `field` is the NOMINAL (un-rescaled) gradient, index == cell position.
  stdout: ONE JSON object (the readout network):
            {"feature": "absolute"|"relative"|"rank",
             "smooth": int >= 0,                 # neighbour radius (clamped to L)
             "cuts":   [c1, c2]}                 # two finite cut points
          Applied per cell: with feature value f, label = 0 if f<c1 else
          1 if f<c2 else 2.

  INVALID output (not a dict, unknown feature, non-integer/negative smooth, cuts not
  a length-2 list of finite numbers), a crash, a timeout, or non-JSON -> that
  instance scores 0.0.

SCORING (deterministic; no wall-time).  For each instance the evaluator builds K
rescaled+corrupted gradients (hidden) with known positional ground truth, applies
the candidate's network to each, and takes the mean fraction of correctly labelled
cells `acc_cand`.  It normalises against the tissue's own trivial baseline
`acc_anchor` = accuracy of painting EVERY cell the majority stripe (label 1, i.e.
0.5 by construction), with a perfect field = 1.0 ideal:
    r = clamp( 0.1 + 0.9 * (acc_cand - acc_anchor) / (1.0 - acc_anchor), 0, 1 )
So do-nothing/majority scores ~0.1; the (unreachable, because of the bump + noise)
perfect labelling scores 1.0 -> genuine headroom.  An absolute-threshold network
calibrated on the nominal field collapses once amplitude/slope are rescaled; a
rank/relative readout that reads RELATIVE position survives.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC (nominal) field.  The rescaled
hidden gradients and the ground-truth labels live only in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _frng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return rnd


# ----------------------------- ground-truth stripes ------------------------
def _target_labels(L):
    q1 = L // 4
    q3 = (3 * L) // 4
    return [0 if i < q1 else (1 if i < q3 else 2) for i in range(L)]


# ----------------------------- instance family -----------------------------
# Each spec: nominal shape (gamma0, A0, B0) + tissue length L + the per-field
# rescaling ranges (amplitude, offset, slope factor, bump strength, noise).
_SPECS = [
    dict(seed=101, L=240, gamma0=1.0, A0=10.0, B0=1.0, K=8, amp=(0.6, 1.7),  off=0.4,  gfac=(0.7, 1.5),  beta=(0.08, 0.16), sigma=0.030),
    dict(seed=102, L=200, gamma0=1.3, A0=8.0,  B0=0.5, K=8, amp=(0.5, 1.8),  off=0.5,  gfac=(0.6, 1.6),  beta=(0.10, 0.18), sigma=0.035),
    dict(seed=103, L=300, gamma0=0.8, A0=12.0, B0=2.0, K=8, amp=(0.6, 1.6),  off=0.3,  gfac=(0.7, 1.4),  beta=(0.08, 0.15), sigma=0.030),
    dict(seed=104, L=260, gamma0=1.5, A0=9.0,  B0=1.0, K=8, amp=(0.5, 1.9),  off=0.5,  gfac=(0.6, 1.7),  beta=(0.10, 0.18), sigma=0.040),
    dict(seed=105, L=220, gamma0=1.0, A0=10.0, B0=0.0, K=8, amp=(0.7, 1.5),  off=0.3,  gfac=(0.8, 1.3),  beta=(0.07, 0.13), sigma=0.025),
    dict(seed=106, L=320, gamma0=1.2, A0=7.0,  B0=1.5, K=8, amp=(0.5, 1.8),  off=0.5,  gfac=(0.6, 1.6),  beta=(0.10, 0.17), sigma=0.035),
    dict(seed=107, L=280, gamma0=0.9, A0=11.0, B0=0.5, K=8, amp=(0.6, 1.7),  off=0.4,  gfac=(0.7, 1.5),  beta=(0.09, 0.16), sigma=0.030),
    # harder / larger held-out instances (wide rescaling -> absolute readout collapses)
    dict(seed=108, L=360, gamma0=1.4, A0=8.5,  B0=1.0, K=8, amp=(0.5, 1.9),  off=0.5,  gfac=(0.6, 1.7),  beta=(0.11, 0.18), sigma=0.040),
    dict(seed=109, L=240, gamma0=1.1, A0=10.0, B0=1.0, K=8, amp=(0.55, 1.75), off=0.45, gfac=(0.65, 1.6), beta=(0.10, 0.17), sigma=0.035),
    dict(seed=110, L=400, gamma0=1.6, A0=9.0,  B0=2.0, K=8, amp=(0.5, 1.9),  off=0.5,  gfac=(0.6, 1.8),  beta=(0.11, 0.19), sigma=0.040),
]


def _build_fields(sp):
    """K rescaled + bump- and noise-corrupted gradients for one instance."""
    seed, L, gamma0, A0, B0, K = sp["seed"], sp["L"], sp["gamma0"], sp["A0"], sp["B0"], sp["K"]
    r = _frng(seed)
    fields = []
    for _ in range(K):
        amp = sp["amp"][0] + (sp["amp"][1] - sp["amp"][0]) * r()
        off = sp["off"] * A0 * r()
        gfac = sp["gfac"][0] + (sp["gfac"][1] - sp["gfac"][0]) * r()
        beta = sp["beta"][0] + (sp["beta"][1] - sp["beta"][0]) * r()
        freq = 1 + int(r() * 2)                 # low-frequency bump: 1 or 2 cycles
        phase = 2 * math.pi * r()
        sigma = sp["sigma"]
        A = A0 * amp
        B = B0 + off
        gamma = gamma0 * gfac
        f = []
        for i in range(L):
            p = i / (L - 1)
            base = B + A * (p ** gamma)
            bump = beta * A * math.sin(2 * math.pi * freq * p + phase)
            noise = sigma * A * (2 * r() - 1)
            f.append(base + bump + noise)
        fields.append(f)
    return fields


def _public_field(sp):
    """Nominal (un-rescaled) gradient shown to the candidate; index == position."""
    L, gamma0, A0, B0, seed = sp["L"], sp["gamma0"], sp["A0"], sp["B0"], sp["seed"]
    r = _frng(seed + 777)
    return [B0 + A0 * ((i / (L - 1)) ** gamma0) + 0.01 * A0 * (2 * r() - 1) for i in range(L)]


# ----------------------------- network application -------------------------
def _is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and x == x and x not in (float("inf"), float("-inf")))


def _smooth(m, w):
    """Neighbour-average with radius w via prefix sums -> O(L) regardless of w."""
    if w <= 0:
        return m
    L = len(m)
    pref = [0.0] * (L + 1)
    for i in range(L):
        pref[i + 1] = pref[i] + m[i]
    out = [0.0] * L
    for i in range(L):
        a = i - w if i - w > 0 else 0
        b = i + w if i + w < L - 1 else L - 1
        out[i] = (pref[b + 1] - pref[a]) / (b - a + 1)
    return out


def _labels_from_net(field, net):
    """Apply a validated network to one field. Returns label list or None."""
    if not isinstance(net, dict):
        return None
    feat = net.get("feature")
    if feat not in ("absolute", "relative", "rank"):
        return None
    w = net.get("smooth", 0)
    if isinstance(w, bool) or not isinstance(w, int) or w < 0:
        return None
    cuts = net.get("cuts")
    if not (isinstance(cuts, list) and len(cuts) == 2 and _is_num(cuts[0]) and _is_num(cuts[1])):
        return None
    c1, c2 = float(cuts[0]), float(cuts[1])
    L = len(field)
    m = _smooth(field, min(w, L))
    if feat == "absolute":
        fv = m
    elif feat == "relative":
        lo = min(m); hi = max(m)
        d = hi - lo if hi - lo > 1e-12 else 1e-12
        fv = [(x - lo) / d for x in m]
    else:  # rank
        order = sorted(range(L), key=lambda i: (m[i], i))
        fv = [0.0] * L
        denom = (L - 1) if L > 1 else 1
        for j, idx in enumerate(order):
            fv[idx] = j / denom
    return [0 if x < c1 else (1 if x < c2 else 2) for x in fv]


def _accuracy(net, fields, tgt):
    tot = 0.0
    n = len(tgt)
    for f in fields:
        lab = _labels_from_net(f, net)
        if lab is None:
            return None
        tot += sum(1 for a, b in zip(lab, tgt) if a == b) / n
    return tot / len(fields)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]

    vec = []
    for sp in _SPECS:
        L = sp["L"]
        tgt = _target_labels(L)
        fields = _build_fields(sp)
        # anchor = majority-stripe (all label 1) accuracy, computed by the parent
        anchor = sum(1 for t in tgt if t == 1) / L
        denom = 1.0 - anchor
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": f"tissue{sp['seed']}", "L": L, "field": list(_public_field(sp))}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            acc_cand = _accuracy(ans, fields, tgt)
        except Exception:
            acc_cand = None
        if acc_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (acc_cand - anchor) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
