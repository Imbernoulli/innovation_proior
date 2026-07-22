import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0931 -- multiscale-error-refinement-allocator (Format B, isolated)
# Theme: adaptive mesh refinement -- distribute a FIXED budget of B piecewise-
# constant reconstruction cells over a 1-D field so that the reconstruction
# error is minimized. The candidate never sees the true field; it only sees
# a coarse "probe" sampling (a cheap pilot solve), from which it must
# ESTIMATE a local error indicator, EQUIDISTRIBUTE the cell budget against
# that indicator (not the raw domain length), and COARSEN over-resolved
# smooth stretches to RECYCLE their cells toward sharp features.
# Objective: MINIMIZE the true reconstruction error. Normalization: the
# uniform-grid reference partition == baseline.
# ==========================================================================

N = 360          # fine atomic grid: true field is defined at integers 0..N-1
N0 = 90          # number of base probe blocks
BW = N // N0     # block width (probe spacing) = 4
B = 18           # fixed total cell budget handed to every instance
assert N % N0 == 0 and N % B == 0 and N0 % B == 0

SEEDS = list(range(93101, 93111))   # 10 fixed, seeded instances
C = 0.055        # normalization constant (trivial/uniform lands near here)


# ---------------------------------------------------------------- field ---
def _make_field(seed):
    """Deterministic smooth background + 2-3 planted fronts/spikes."""
    rng = random.Random(seed)
    bg_level = rng.uniform(-1.0, 1.0)
    bg_slope = rng.uniform(-0.002, 0.002)
    bg_amp = rng.uniform(0.05, 0.25)
    bg_cycles = rng.choice([1, 2, 3])
    nfeat = rng.choice([2, 2, 3])
    feats = []
    for _ in range(nfeat):
        kind = rng.choice(["front", "spike"])
        center = rng.uniform(20, N - 20)
        if kind == "front":
            width = rng.uniform(6, 14)
        else:
            width = rng.uniform(7, 18)
        amp = rng.uniform(1.0, 6.0) * rng.choice([-1, 1])
        feats.append((kind, center, width, amp))

    def F(x):
        v = bg_level + bg_slope * x + bg_amp * math.sin(2 * math.pi * bg_cycles * x / N)
        for kind, c, w, a in feats:
            if kind == "front":
                v += a * (1.0 / (1.0 + math.exp(-(x - c) / w)) - 0.5)
            else:
                v += a * math.exp(-((x - c) / w) ** 2)
        return v
    return F


def _fine_array(F):
    return [F(i) for i in range(N)]


def _probe_arrays(F):
    xs = [j * BW for j in range(N0 + 1)]
    fs = [F(x) for x in xs]
    return xs, fs


# --------------------------------------------------------------- scoring ---
def _total_err(fine, cuts):
    """cuts: sorted real boundaries strictly inside (0,N). Reconstructs each
    resulting cell by its TRUE mean (an oracle least-squares fit given the
    candidate's chosen cell placement) and sums squared deviations."""
    bnds = [0.0] + list(cuts) + [float(N)]
    tot = 0.0
    for k in range(len(bnds) - 1):
        lo, hi = bnds[k], bnds[k + 1]
        # half-open cell [lo, hi): an integer point p belongs here iff lo <= p < hi.
        # math.ceil already implements this exactly for both integer- and real-
        # valued boundaries (no epsilon fudge needed / wanted -- see fsx_B_0931
        # r1 review: a -1e-9 tolerance here silently reassigns points that sit
        # within 1e-9 of an integer boundary to the wrong cell).
        lo_i = math.ceil(lo)
        hi_i = math.ceil(hi)
        members = fine[lo_i:hi_i]
        if not members:
            continue
        m = sum(members) / len(members)
        tot += sum((v - m) ** 2 for v in members)
    return tot


def _uniform_cuts():
    step = N // B
    return [step * k for k in range(1, B)]


def make_instances():
    out = []
    for seed in SEEDS:
        F = _make_field(seed)
        fine = _fine_array(F)
        xs, fs = _probe_arrays(F)
        # NOTE: the generator seed is intentionally NOT exposed in the public
        # instance -- it deterministically reconstructs the exact hidden field
        # via _make_field(), so publishing it would let a candidate that knows
        # (or reimplements) that formula solve the true field directly instead
        # of estimating an indicator from probe_f, bypassing the intended task
        # entirely (fsx_B_0931 r1 review). Solvers get probe_x/probe_f only.
        pub = {
            "N": N, "N0": N0, "B": B, "BW": BW,
            "probe_x": xs, "probe_f": [round(v, 8) for v in fs],
        }
        out.append({"public": pub, "hidden": {"fine": fine}})
    return out


def baseline(inst):
    fine = inst["hidden"]["fine"]
    return _total_err(fine, _uniform_cuts())


def score(inst, ans):
    fine = inst["hidden"]["fine"]
    if not isinstance(ans, dict) or "cuts" not in ans:
        return False, 0.0
    cuts = ans["cuts"]
    if not isinstance(cuts, list) or len(cuts) != B - 1:
        return False, 0.0
    clean = []
    for v in cuts:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, 0.0
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return False, 0.0
        if not (0.0 < v < float(N)):
            return False, 0.0
        clean.append(v)
    for a, b in zip(clean, clean[1:]):
        if not (a < b):
            return False, 0.0
    err = _total_err(fine, clean)
    if not (err == err) or err < 0.0 or not math.isfinite(err):
        return False, 0.0
    return True, err


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, C * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
