# FROZEN evaluator -- Format B, isolated. Do not edit to fit a candidate.
# Task: geothermal-well completion-log validity classifier under a fixed linear model class,
# scored on IN-DISTRIBUTION (short) plus LENGTH-OOD (longer) well logs.  The candidate never
# sees the hidden test logs -- only a menu of order-invariant "bag" features on a small TRAIN
# set -- and outputs weights w,b for the frozen scorer  pred = 1[ w.f + b > 0 ].
import sys, json, math, random, isorun

# ---- geothermal-well "typed-Dyck" instance generator (deterministic) --------------------
# token = (kind, casing_type): kind 0 = set/open casing, kind 1 = pull/seal casing.
# A log is VALID iff it is a proper typed-Dyck word (every seal matches the most recent
# unsealed casing OF THE SAME size) AND the nesting depth never exceeds Dmax (rig limit).

FEATURE_NAMES = ["abs_balance", "max_depth", "mean_depth", "length", "n_open", "open_frac"]
M = len(FEATURE_NAMES)


def compute_features(seq):
    n = len(seq)
    n_open = sum(1 for k, t in seq if k == 0)
    n_close = n - n_open
    bal = 0
    mx = 0
    tot = 0
    for k, t in seq:
        bal += 1 if k == 0 else -1
        if bal > mx:
            mx = bal
        tot += bal
    length = float(n)
    abs_balance = float(abs(n_open - n_close))
    max_depth = float(mx)
    mean_depth = tot / n if n else 0.0
    open_frac = n_open / n if n else 0.0
    return [abs_balance, max_depth, mean_depth, length, float(n_open), open_frac]


def gen_valid(rng, p, Dmax, K):
    """A proper typed-Dyck word of p pairs with nesting depth <= Dmax (VALID)."""
    seq = []
    stack = []
    opens_left = p
    while opens_left > 0 or stack:
        can_open = opens_left > 0 and len(stack) < Dmax
        can_close = len(stack) > 0
        if can_open and (not can_close or len(stack) == 0 or rng.random() < 0.5):
            t = rng.randrange(K)
            stack.append(t)
            seq.append((0, t))
            opens_left -= 1
        elif can_close:
            t = stack.pop()
            seq.append((1, t))
        else:  # opens_left==0 and stack empty
            break
    return seq


def gen_deep(rng, p, Dmax, K):
    """Balanced, properly typed, but nesting depth > Dmax  -> INVALID (visible via max_depth)."""
    d = min(p, Dmax + 1 + rng.randrange(2))
    types = [rng.randrange(K) for _ in range(d)]
    seq = [(0, t) for t in types] + [(1, t) for t in reversed(types)]
    for _ in range(p - d):  # shallow padding keeps depth == d
        t = rng.randrange(K)
        seq += [(0, t), (1, t)]
    return seq


def gen_imbalance(rng, p, Dmax, K):
    """A valid word plus a dangling open -> unmatched casing (INVALID, visible via abs_balance)."""
    seq = gen_valid(rng, p, Dmax, K)
    seq = seq + [(0, rng.randrange(K))]
    return seq


def gen_typeswap(rng, p, Dmax, K):
    """A valid word with one seal re-labelled to the WRONG casing size -> mismatch (INVALID).
    Only the casing TYPE of a seal changes; every provided feature is type-agnostic, so this
    corruption is INVISIBLE to any bag-feature model -> irreducible error floor (headroom)."""
    seq = gen_valid(rng, p, Dmax, K)
    close_idx = [i for i, (k, t) in enumerate(seq) if k == 1]
    if not close_idx:
        return seq + [(0, rng.randrange(K))]  # degenerate fallback -> imbalance
    j = rng.choice(close_idx)
    k, t = seq[j]
    t2 = (t + 1 + rng.randrange(K - 1)) % K
    seq[j] = (1, t2)
    return seq


def make_bucket(rng, n_side, p_pos_lo, p_pos_hi, p_neg_lo, p_neg_hi, Dmax, K):
    rows = []  # (features, label)
    for _ in range(n_side):  # positives (valid)
        p = rng.randint(p_pos_lo, p_pos_hi)
        rows.append((compute_features(gen_valid(rng, p, Dmax, K)), 1))
    gens = [gen_imbalance, gen_deep, gen_typeswap]
    for i in range(n_side):  # negatives (invalid), balanced across the 3 corruption modes
        p = rng.randint(p_neg_lo, p_neg_hi)
        rows.append((compute_features(gens[i % 3](rng, p, Dmax, K)), 0))
    rng.shuffle(rows)
    return rows


def make_instances():
    out = []
    for s in range(10):
        rng = random.Random(1000 + 7 * s)
        K = 2 + (s % 2)          # 2 or 3 casing sizes
        Dmax = 4 + (s % 3)       # rig depth limit 4..6
        # ID buckets: negatives run LONGER than positives -> `length` is spuriously predictive
        # IN-DISTRIBUTION.  OOD buckets: both classes long, no length gap -> length overfitting dies.
        train = make_bucket(rng, 30, 5, 10, 11, 17, Dmax, K)
        test_id = make_bucket(rng, 30, 5, 10, 11, 17, Dmax, K)
        ood_med = make_bucket(rng, 30, 20, 28, 20, 28, Dmax, K)
        ood_long = make_bucket(rng, 30, 36, 46, 36, 46, Dmax, K)
        public = {
            "m": M,
            "feature_names": FEATURE_NAMES,
            "Dmax": Dmax,
            "K": K,
            "train": [[*f, lab] for (f, lab) in train],
        }
        hidden = {"buckets": {"id": test_id, "ood_med": ood_med, "ood_long": ood_long}}
        out.append({"public": public, "hidden": hidden})
    return out


def _gmean(xs):
    acc = 0.0
    for x in xs:
        acc += math.log(max(x, 1e-9))
    return math.exp(acc / len(xs))


def _bucket_acc(rows, predict):
    c = 0
    for f, lab in rows:
        if predict(f) == lab:
            c += 1
    return c / len(rows)


def baseline(inst):
    # trivial construction: always predict VALID (=1). Its error is the calibration point.
    buckets = inst["hidden"]["buckets"]
    accs = [_bucket_acc(rows, lambda f: 1) for rows in buckets.values()]
    return 1.0 - _gmean(accs)


def score(inst, ans):
    m = inst["public"]["m"]
    if not isinstance(ans, dict):
        return False, 0.0
    w = ans.get("w")
    b = ans.get("b")
    if not isinstance(w, list) or len(w) != m:
        return False, 0.0
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        return False, 0.0
    try:
        wf = [float(x) for x in w]
        bf = float(b)
    except (TypeError, ValueError):
        return False, 0.0
    if any(not math.isfinite(x) for x in wf) or not math.isfinite(bf):
        return False, 0.0

    def predict(f):
        s = bf
        for wi, fi in zip(wf, f):
            s += wi * fi
        return 1 if s > 0.0 else 0

    buckets = inst["hidden"]["buckets"]
    accs = [_bucket_acc(rows, predict) for rows in buckets.values()]
    err = 1.0 - _gmean(accs)   # minimize error
    if not math.isfinite(err):
        return False, 0.0
    return True, err


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, err = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(err, 1e-9))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    print("Ratio: %.6f" % (sum(vec) / len(vec)))
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
