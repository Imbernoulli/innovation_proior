import sys, json, math, random
from collections import Counter
import isorun

# ============================================================================
# Telescope Array Focus Calibration  (Format B, isolated, deterministic)
#
# Each "telescope array" has a HIDDEN local calibration law: the corrected
# focus state y[i] of mirror i depends on the raw states of mirror i and its
# two upstream neighbours via a hidden order-3 lookup table T3:
#       y[i] = T3[ x[i-2], x[i-1], x[i] ]      (x[<0] = boundary state)
# The law is purely LOCAL / length-invariant, so it applies unchanged to
# arrays of ANY length.  The candidate is given labelled (x,y) training arrays
# of an in-distribution length, and must predict y for query arrays of both the
# in-distribution length (ID) and a longer out-of-distribution length (OOD).
# Score per instance = gmean(token_accuracy_ID, token_accuracy_OOD), which
# rewards a length-generalising inductive bias over absolute-position memoing.
# The hidden table + query targets never leave the parent process.
# ============================================================================

# 8 instances: (K alphabet, n_train, L_id, L_ood).  Longer OOD arrays and
# larger alphabets are the harder / held-out generalization cases.
CONFIGS = [(6, 40, 20, 44), (6, 40, 20, 60), (7, 40, 20, 44), (7, 44, 22, 60),
           (6, 36, 18, 50), (7, 40, 20, 52), (6, 44, 22, 48), (7, 36, 20, 56)]
N_QUERY = 8


def _make_one(seed, K, n_train, L_id, L_ood, n_q):
    rng = random.Random(seed)
    boundary = 0
    T3 = {}                                   # hidden order-3 calibration law
    def law(a, b, c):
        k = (a, b, c)
        if k not in T3:
            T3[k] = rng.randrange(K)
        return T3[k]
    # skewed (Zipf-ish) raw-state distribution: common contexts are dense,
    # rare contexts are sparse -> lower-order models capture the dominant
    # contexts, full coverage stays out of reach (headroom).
    w = [1.0 / (j + 1) for j in range(K)]
    sw = sum(w)
    cum = []
    c = 0.0
    for x in w:
        c += x / sw
        cum.append(c)
    def draw():
        u = rng.random()
        for i, cc in enumerate(cum):
            if u <= cc:
                return i
        return K - 1
    def gen(L):
        x = [draw() for _ in range(L)]
        y = []
        for i in range(L):
            a = x[i - 2] if i > 1 else boundary
            b = x[i - 1] if i > 0 else boundary
            y.append(law(a, b, x[i]))
        return x, y
    train = [gen(L_id) for _ in range(n_train)]
    q_id = [gen(L_id) for _ in range(n_q)]
    q_ood = [gen(L_ood) for _ in range(n_q)]
    public = {
        "K": K,
        "boundary": boundary,
        "train": [{"x": x, "y": y} for x, y in train],
        "queries": {
            "id": [x for x, _ in q_id],
            "ood": [x for x, _ in q_ood],
        },
    }
    hidden = {
        "id": [y for _, y in q_id],
        "ood": [y for _, y in q_ood],
    }
    return {"public": public, "hidden": hidden}


def make_instances():
    return [_make_one(5000 + i, *cfg, N_QUERY) for i, cfg in enumerate(CONFIGS)]


def _acc(pred_group, true_group):
    tot = 0
    cor = 0
    for ps, ts in zip(pred_group, true_group):
        for a, b in zip(ps, ts):
            tot += 1
            cor += 1 if a == b else 0
    return cor / tot if tot else 0.0


def _gmean(a, b):
    return math.sqrt(max(a, 1e-6) * max(b, 1e-6))


def baseline(inst):
    # trivial construction the evaluator computes ITSELF: predict the single
    # most-frequent corrected state seen in training (a constant array).
    g = Counter()
    for tr in inst["public"]["train"]:
        for v in tr["y"]:
            g[v] += 1
    m = g.most_common(1)[0][0] if g else 0
    pub = inst["public"]["queries"]
    hid = inst["hidden"]
    p_id = [[m] * len(x) for x in pub["id"]]
    p_ood = [[m] * len(x) for x in pub["ood"]]
    return _gmean(_acc(p_id, hid["id"]), _acc(p_ood, hid["ood"]))


def _valid_group(pred, ref, K):
    if not isinstance(pred, list) or len(pred) != len(ref):
        return False
    for ps, rs in zip(pred, ref):
        if not isinstance(ps, list) or len(ps) != len(rs):
            return False
        for v in ps:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= K:
                return False
    return True


def score(inst, ans):
    K = inst["public"]["K"]
    hid = inst["hidden"]
    if not isinstance(ans, dict):
        return False, 0.0
    pred = ans.get("predictions")
    if not isinstance(pred, dict):
        return False, 0.0
    pid = pred.get("id")
    pood = pred.get("ood")
    if not _valid_group(pid, hid["id"], K):
        return False, 0.0
    if not _valid_group(pood, hid["ood"], K):
        return False, 0.0
    obj = _gmean(_acc(pid, hid["id"]), _acc(pood, hid["ood"]))
    if not (obj == obj and obj < math.inf):
        return False, 0.0
    return True, obj


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
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))   # maximization: trivial->0.1
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    print("Ratio: %.6f" % (sum(vec) / len(vec)))
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
