import sys, json, math, random
from collections import Counter
import isorun

# ============================================================================
# Harbor Container Port -- Berth-Code Prediction (Dyck length-generalization)
#
# A single quay crane executes a MANIFEST of moves on ONE LIFO stack of
# containers.  A move is either:
#     LOAD(t)  -- place a container of bay-type t (0..K-1) on top of the stack,
#     UNLOAD   -- remove the top container.
# A manifest is a well-nested (balanced) Dyck-K word: no UNLOAD ever hits an
# empty stack and the stack is empty at the end.
#
# Whenever a container is UNLOADED the port assigns it a HIDDEN "berth code"
# in 0..C-1.  The code is a hidden LOCAL law of the removed container's bay-type
# (`top`) AND the bay-type of the container it was resting on (`under`, = the
# newly exposed top, or the special "quay" value K when the stack becomes
# empty):
#         code = HIDDEN[(top, under)]              (deterministic, length-free)
#
# To predict a code you must (a) correctly track the LIFO stack to know top/under
# at each UNLOAD -- an algorithmic, length-generalizing skill -- and (b) infer
# the hidden (top,under) -> code table from labelled training manifests.  The
# candidate sees labelled training manifests of an in-distribution length and
# must predict UNLOAD codes for query manifests at both the in-distribution
# length (ID) and a LONGER out-of-distribution length (OOD).  Longer manifests
# reach DEEPER stacks -> more rare/unseen (top,under) pairs, so full coverage
# stays out of reach (headroom) and the score rewards a stack-based inductive
# bias over positional memorization.  Score per instance = gmean(acc_ID,acc_OOD)
# over UNLOAD positions.  The hidden table + query codes never leave the parent.
# ============================================================================

# 8 instances: (K bay-types, n_train, npairs_id, npairs_ood, C berth codes).
# Larger K and longer OOD manifests are the harder / held-out cases.
CONFIGS = [
    (4, 40, 12, 26, 4),
    (4, 40, 12, 34, 4),
    (5, 40, 12, 26, 5),
    (5, 44, 13, 34, 5),
    (4, 36, 11, 30, 4),
    (5, 40, 12, 32, 5),
    (4, 44, 13, 28, 4),
    (5, 36, 12, 34, 5),
]
N_QUERY = 8


def _make_one(seed, K, n_train, npairs_id, npairs_ood, C, n_q):
    rng = random.Random(seed)
    QUAY = K                                    # "under" value when stack empties

    HIDDEN = {}                                 # hidden (top, under) -> berth code
    def law(top, under):
        k = (top, under)
        if k not in HIDDEN:
            HIDDEN[k] = rng.randrange(C)
        return HIDDEN[k]

    # skewed (Zipf-ish) bay-type distribution: common types are dense, rare
    # types sparse -> low-order (top-only) models capture the dominant mass but
    # full (top,under) coverage stays out of reach.
    w = [1.0 / (j + 1) for j in range(K)]
    sw = sum(w)
    cum = []
    acc = 0.0
    for x in w:
        acc += x / sw
        cum.append(acc)
    def draw_type():
        u = rng.random()
        for i, cc in enumerate(cum):
            if u <= cc:
                return i
        return K - 1

    def gen(npairs):
        # Emit a valid balanced Dyck-K manifest of exactly 2*npairs moves.
        # moves: int >=0 = LOAD(type); -1 = UNLOAD.
        moves = []
        codes = []                              # aligned to UNLOAD positions
        stack = []
        loads_done = 0
        total = 2 * npairs
        for _ in range(total):
            can_load = loads_done < npairs
            can_unload = len(stack) > 0
            if can_load and can_unload:
                do_load = rng.random() < 0.5
            else:
                do_load = can_load
            if do_load:
                t = draw_type()
                stack.append(t)
                loads_done += 1
                moves.append(t)
            else:
                top = stack.pop()
                under = stack[-1] if stack else QUAY
                moves.append(-1)
                codes.append(law(top, under))
        return moves, codes

    train = [gen(npairs_id) for _ in range(n_train)]
    q_id = [gen(npairs_id) for _ in range(n_q)]
    q_ood = [gen(npairs_ood) for _ in range(n_q)]

    public = {
        "K": K,
        "C": C,
        "quay": QUAY,
        "train": [{"moves": m, "codes": c} for m, c in train],
        "queries": {
            "id": [m for m, _ in q_id],
            "ood": [m for m, _ in q_ood],
        },
    }
    hidden = {
        "id": [c for _, c in q_id],
        "ood": [c for _, c in q_ood],
    }
    return {"public": public, "hidden": hidden}


def make_instances():
    return [_make_one(7000 + i, *cfg, N_QUERY) for i, cfg in enumerate(CONFIGS)]


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
    # most-frequent berth code seen in training, for every UNLOAD.
    g = Counter()
    for tr in inst["public"]["train"]:
        for v in tr["codes"]:
            g[v] += 1
    m = g.most_common(1)[0][0] if g else 0
    hid = inst["hidden"]
    p_id = [[m] * len(c) for c in hid["id"]]
    p_ood = [[m] * len(c) for c in hid["ood"]]
    return _gmean(_acc(p_id, hid["id"]), _acc(p_ood, hid["ood"]))


def _valid_group(pred, ref, C):
    if not isinstance(pred, list) or len(pred) != len(ref):
        return False
    for ps, rs in zip(pred, ref):
        if not isinstance(ps, list) or len(ps) != len(rs):
            return False
        for v in ps:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= C:
                return False
    return True


def score(inst, ans):
    C = inst["public"]["C"]
    hid = inst["hidden"]
    if not isinstance(ans, dict):
        return False, 0.0
    pred = ans.get("predictions")
    if not isinstance(pred, dict):
        return False, 0.0
    pid = pred.get("id")
    pood = pred.get("ood")
    if not _valid_group(pid, hid["id"], C):
        return False, 0.0
    if not _valid_group(pood, hid["ood"], C):
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
        r = min(1.0, 0.1 * obj / max(b, 1e-12))   # maximization: trivial -> 0.1
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    print("Ratio: %.6f" % (sum(vec) / len(vec)))
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
