# ============================================================================
# SHARED CORE (byte-identical block pasted into gen.py AND verify.py).
# Do NOT import this file directly from either -- it is a source-of-truth
# staging copy only; the harness must not see an importable ground-truth
# module sitting in the problem directory.
# ============================================================================
import math
import random

T0 = 20.0  # fixed ambient / cold-start temperature, same for every job

# Duration ranges below are MULTIPLES of the hidden time constant
# (tau_ref = geometric mean of 1/k_h, 1/k_c) -- NOT absolute numbers -- so
# nothing about a job's physical timescale is guessable from the ladder
# alone; only reading the training tickets reveals what timescale matters.
LADDER = {
    1:  dict(N=40,  Q=8,  dtr_m=(0.35, 2.2),  dqu_m=(0.02, 0.20), cftr=0.30, cfqu=0.65),
    2:  dict(N=50,  Q=10, dtr_m=(0.35, 2.4),  dqu_m=(0.02, 0.22), cftr=0.30, cfqu=0.65),
    3:  dict(N=60,  Q=10, dtr_m=(0.30, 2.4),  dqu_m=(0.02, 0.20), cftr=0.35, cfqu=0.70),
    4:  dict(N=70,  Q=12, dtr_m=(0.30, 2.6),  dqu_m=(0.015, 0.22), cftr=0.35, cfqu=0.70),
    5:  dict(N=90,  Q=14, dtr_m=(0.28, 2.6),  dqu_m=(0.015, 0.20), cftr=0.40, cfqu=0.75),
    6:  dict(N=100, Q=14, dtr_m=(0.28, 2.8),  dqu_m=(0.015, 0.22), cftr=0.40, cfqu=0.75),
    7:  dict(N=130, Q=16, dtr_m=(0.25, 2.8),  dqu_m=(0.012, 0.20), cftr=0.45, cfqu=0.80),
    8:  dict(N=150, Q=18, dtr_m=(0.25, 3.0),  dqu_m=(0.012, 0.22), cftr=0.45, cfqu=0.80),
    9:  dict(N=180, Q=20, dtr_m=(0.22, 3.0),  dqu_m=(0.010, 0.20), cftr=0.50, cfqu=0.85),
    10: dict(N=220, Q=24, dtr_m=(0.22, 3.2),  dqu_m=(0.010, 0.22), cftr=0.50, cfqu=0.85),
}

S_LO, S_HI = -20.0, 260.0
SQ_LO, SQ_HI = -60.0, 320.0   # query setpoints extrapolate beyond the training convex hull
W_LO, W_HI = 12.0, 36.0


def seg_end(S, Tstart, k, D):
    """Closed-form first-order lag: temperature after duration D holding setpoint S,
    starting from Tstart, at rate k."""
    return S + (Tstart - S) * math.exp(-k * D)


def rate_for(S, Tstart, k_h, k_c):
    """Asymmetric law: heating rate k_h applies when moving UP toward S, cooling
    rate k_c when moving DOWN. Sign is fixed for the whole segment (S, Tstart fixed)."""
    return k_h if S >= Tstart else k_c


def eval_job(S1, D1, S2, D2, w, Lo, Hi, k_h, k_c):
    """Simulate a two-segment schedule (ramp then hold) and return the checked
    margin against the window [Lo,Hi] over the final w seconds of the hold."""
    if D1 > 0:
        k1 = rate_for(S1, T0, k_h, k_c)
        T_mid = seg_end(S1, T0, k1, D1)
    else:
        T_mid = T0
    k2 = rate_for(S2, T_mid, k_h, k_c)
    checkStart = D2 - w
    Ta = seg_end(S2, T_mid, k2, checkStart)
    Tb = seg_end(S2, T_mid, k2, D2)
    lo_t, hi_t = (Ta, Tb) if Ta <= Tb else (Tb, Ta)
    slack_lo = lo_t - Lo
    slack_hi = Hi - hi_t
    margin = min(slack_lo, slack_hi)
    nm = margin / ((Hi - Lo) / 2.0)
    label = 1 if margin >= 0.0 else 0
    return margin, nm, label


def rand_job(rng, dmult, tau_ref, compound_frac, srange=(S_LO, S_HI)):
    compound = rng.random() < compound_frac
    S2 = rng.uniform(*srange)
    D2 = rng.uniform(*dmult) * tau_ref
    W = rng.uniform(W_LO, W_HI)
    jitter = rng.uniform(-0.5, 0.5) * W
    center = S2 + jitter
    Lo = center - W / 2.0
    Hi = center + W / 2.0
    w = 0.3 * D2
    if compound:
        S1 = rng.uniform(*srange)
        D1 = rng.uniform(*dmult) * tau_ref
    else:
        S1, D1 = T0, 0.0
    return S1, D1, S2, D2, w, Lo, Hi


def straddle_pair(rng, k_h, k_c, heating):
    """Pure hold job (D1=0) isolating a single rate. Since the check window's FAR-
    from-target endpoint (checkStart) is the binding constraint whenever S2 sits
    inside the window, solve checkStart's time in closed form so it lands EXACTLY
    on the binding boundary (Lo for a heating approach, Hi for a cooling one), then
    emit BOTH a just-pass and a just-fail perturbation of that same instant -- a
    genuine two-sided bracket on the rate constant, planted for identifiability."""
    frac = 0.3
    k = k_h if heating else k_c
    S2 = rng.uniform(120.0, 220.0) if heating else rng.uniform(-20.0, -5.0)
    W = rng.uniform(W_LO, W_HI)
    offset = rng.uniform(0.0, 0.15) * W
    center = (S2 - offset) if heating else (S2 + offset)
    Lo = center - W / 2.0
    Hi = center + W / 2.0
    target = Lo if heating else Hi
    ratio = (target - S2) / (T0 - S2)
    if not (0.0 < ratio < 1.0):
        return None
    t0 = -math.log(ratio) / k
    D2 = t0 / (1.0 - frac)
    if D2 <= 1e-6:
        return None
    eps = rng.uniform(0.02, 0.05) * D2
    out = []
    for sign in (-1, 1):
        D2p = D2 + sign * eps
        if D2p <= 1e-6:
            continue
        w = frac * D2p
        out.append((T0, 0.0, S2, D2p, w, Lo, Hi))
    return out if len(out) == 2 else None


def build_jobs(rng, n, dmult, tau_ref, compound_frac, k_h, k_c, n_pairs):
    jobs = []
    ci = 0
    tries = 0
    while len(jobs) < n_pairs * 2 and tries < 60 * n_pairs:
        tries += 1
        heating = (ci % 2 == 0)
        ci += 1
        pair = straddle_pair(rng, k_h, k_c, heating)
        if pair is not None:
            jobs.extend(pair)
    while len(jobs) < n:
        jobs.append(rand_job(rng, dmult, tau_ref, compound_frac))
    rng.shuffle(jobs)
    return jobs


def draw_rates(testId):
    """Wide, log-uniform, hard-to-guess draw for the two hidden rate constants.
    A rejection loop keeps a genuine heating/cooling asymmetry (k_h >= 1.3*k_c)."""
    rng_p = random.Random(300000 + 17 * testId)
    k_h = k_c = None
    for _ in range(200):
        k_h = math.exp(rng_p.uniform(math.log(0.03), math.log(1.5)))
        k_c = math.exp(rng_p.uniform(math.log(0.003), math.log(0.4)))
        if k_h >= 1.3 * k_c:
            return k_h, k_c
    return k_h, k_c


def hidden_construct(testId):
    """Deterministically (re)build the hidden asymmetric rates (k_h, k_c), the
    training tickets, and the held-out query list for a given testId. Identical
    in gen.py and verify.py -- NEVER import this across files; the checker must
    not depend on gen.py's presence."""
    cfg = LADDER[testId]
    k_h, k_c = draw_rates(testId)
    tau_ref = math.sqrt((1.0 / k_h) * (1.0 / k_c))
    rng_tr = random.Random(400000 + 31 * testId)
    n_pairs = max(8, cfg['N'] // 10)
    train_specs = build_jobs(rng_tr, cfg['N'], cfg['dtr_m'], tau_ref, cfg['cftr'], k_h, k_c, n_pairs)
    train = []
    for (S1, D1, S2, D2, w, Lo, Hi) in train_specs:
        _, _, label = eval_job(S1, D1, S2, D2, w, Lo, Hi, k_h, k_c)
        train.append((S1, D1, S2, D2, w, Lo, Hi, label))
    rng_qu = random.Random(500000 + 37 * testId)
    query_specs = [rand_job(rng_qu, cfg['dqu_m'], tau_ref, cfg['cfqu'], (SQ_LO, SQ_HI)) for _ in range(cfg['Q'])]
    query = []
    for (S1, D1, S2, D2, w, Lo, Hi) in query_specs:
        _, nm, label = eval_job(S1, D1, S2, D2, w, Lo, Hi, k_h, k_c)
        query.append((S1, D1, S2, D2, w, Lo, Hi, nm, label))
    return dict(N=cfg['N'], Q=cfg['Q'], k_h=k_h, k_c=k_c, tau_ref=tau_ref, train=train, query=query)
# ============================================================================
# END SHARED CORE
# ============================================================================


import re

CAP_Q = 4.0        # per-query clip on |pred-true| normalized-margin error
RATIO_CAP = 900.0  # ratio numerator cap -> best-possible score is 0.9 (headroom kept above it)

FLOAT_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def parse_floats(path, need):
    try:
        txt = open(path, "r").read()
    except Exception:
        return None
    toks = txt.split()
    if len(toks) != need:
        return None
    vals = []
    for t in toks:
        if not FLOAT_RE.match(t):
            return None
        try:
            v = float(t)
        except ValueError:
            return None
        if not math.isfinite(v):
            return None
        vals.append(v)
    return vals


def main():
    import sys
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        toks = f.read().split()
    it = iter(toks)
    testId = int(next(it)); N = int(next(it)); Q = int(next(it))
    for _ in range(N):
        for _ in range(7):
            next(it)
        next(it)  # label
    shown_queries = []
    for _ in range(Q):
        shown_queries.append(tuple(float(next(it)) for _ in range(7)))

    inst = hidden_construct(testId)
    if inst['N'] != N or inst['Q'] != Q:
        print("Ratio: 0.0  # instance metadata mismatch")
        return
    # Score against the EXACT (rounded) numbers the solver was shown, re-evaluated
    # with the recovered hidden rates -- not the unrounded internal floats -- so a
    # perfect physics recovery from the printed schedule can never be penalized by
    # print-precision drift.
    true_nms = [eval_job(*sq, inst['k_h'], inst['k_c'])[1] for sq in shown_queries]

    preds = parse_floats(outf, Q)
    if preds is None:
        print("Ratio: 0.0  # malformed output: need exactly %d finite base-10 floats" % Q)
        return

    losses, blosses = [], []
    for pred, true_nm in zip(preds, true_nms):
        losses.append(min(CAP_Q, abs(pred - true_nm)))
        blosses.append(min(CAP_Q, abs(0.0 - true_nm)))
    F = sum(losses) / len(losses)
    B = sum(blosses) / len(blosses)
    sc = min(RATIO_CAP, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
