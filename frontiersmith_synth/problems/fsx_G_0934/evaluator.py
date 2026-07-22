#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0934 -- "Small Automaton, Big Generalization: Inducing a
DFA from Labeled Strings" (family: rpni-dfa-induction; format B, quality-metric).

THEME.  A protocol-analysis team logs traces of interactions with black-box binary-
alphabet devices.  Each device is secretly driven by a SMALL deterministic finite
automaton (DFA).  The team observes a finite labeled TRAINING sample (strings over
{0,1} with an accept/reject outcome) and must submit a DFA that reproduces the
device's behaviour on a HELD-OUT set of traces it never sees -- some considerably
longer than anything in the training log.

TASK (the model writes a DFA-induction heuristic).  Given ONLY the public training
sample, the candidate outputs a DFA (states, complete transition function, start
state, accepting states).  The evaluator runs that DFA on the hidden held-out traces
of the SAME device and scores accuracy, blended with a compactness term (a smaller
automaton that matches behaviour scores higher than a bloated one that happens to
also get the labels right) -- rewarding genuine generalization/compression, not raw
memorization size.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : {"name": str, "alphabet": ["0","1"],
           "train": [{"s": "0110", "label": 0/1}, ...]}
  stdout: {"delta": [[n0,n1], ...], "start": int, "accept": [int, ...]}
  VALID iff delta is a non-empty list of 2-int rows whose entries are all state
  indices in range, start is a valid index, and every accept entry is a valid index.
  Anything else (crash, timeout, non-JSON, null, wrong type, out-of-range index) ->
  that device scores 0.0.

TRAP (mechanism: the obvious approach).  The literal prefix-tree acceptor (PTA) --
one state per distinct prefix seen in training, completed with a reject sink for
every unseen continuation -- gets EVERY training example exactly right but has
(roughly) as many states as the training sample has distinct prefixes, and any
held-out trace that runs off the memorized prefixes is rejected by default.  Because
several devices in this bank are tested on traces markedly longer than the training
log, the PTA's accuracy collapses on those devices while a compact automaton that
actually captures the device's regularity keeps predicting correctly.

INSIGHT (mechanism: evidence-based state merging).  Two prefixes that LOOK distinct
in the tree may secretly reach the same underlying device state.  RPNI-style merging
folds prefixes together whenever doing so stays consistent with every labeled
training example (never merging an evidenced-accept prefix with an evidenced-reject
one, propagating merges through shared transitions).  This recovers a small
automaton that predicts unseen, longer traces -- the prefix tree cannot.

SCORING (deterministic; see score()).  Per device:
  a_triv   = accuracy of the majority-class rule on the held-out set   (weak anchor)
  a_oracle = accuracy of the TRUE device automaton on the held-out set (strong anchor,
             always == 1.0 by construction since it generated the labels)
  a_cand   = accuracy of the candidate's DFA on the held-out set
  s        = number of the candidate's states REACHABLE from its start state
  m        = the TRUE device's minimal state count (never revealed to the candidate)
  acc_r  = clamp((a_cand - a_triv) / max(a_oracle - a_triv, 0.05), 0, 1)
  size_r = m / max(s, m)
  r      = clamp(0.1 + 0.75 * acc_r * (0.5 + 0.5 * size_r), 0, 1)
Ratio = mean(r) over the device bank.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS
via `isorun.run_candidate`; it only ever sees the public training sample.  The held-
out traces, the true device automaton, and every scoring anchor are computed by THIS
parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean of r over all devices, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

MAX_STATES = 4000
EPS_ACC = 0.05


# ----------------------------- deterministic RNG (splitmix64) --------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def _next():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        z = z ^ (z >> 31)
        return z

    def u():
        return (_next() & ((1 << 53) - 1)) / float(1 << 53)

    def randint(a, b):
        return a + int(u() * (b - a + 1))

    return u, randint


# ----------------------------- target automaton -----------------------------
def _build_target(seed, k):
    """A random, fully-reachable, complete DFA over {0,1} with k states."""
    u, randint = _rng(seed)
    delta = [[None, None] for _ in range(k)]
    # guarantee every state 1..k-1 is reachable from state 0
    for i in range(1, k):
        j = randint(0, i - 1)
        sym = randint(0, 1)
        delta[j][sym] = i
    for i in range(k):
        for sym in (0, 1):
            if delta[i][sym] is None:
                delta[i][sym] = randint(0, k - 1)
    n_accept = max(1, min(k - 1, k // 2))
    perm = list(range(k))
    for i in range(k - 1, 0, -1):
        j = randint(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    accept = set(perm[:n_accept])
    return delta, 0, accept, u, randint


def _run_dfa(delta, start, accept, s):
    st = start
    for ch in s:
        st = delta[st][0 if ch == "0" else 1]
    return 1 if st in accept else 0


def _minimize_count(delta, accept, k):
    """Moore partition refinement -> the number of behaviorally distinct states."""
    part = [1 if i in accept else 0 for i in range(k)]
    for _ in range(k + 1):
        sig_map = {}
        newpart = [0] * k
        for i in range(k):
            sig = (part[i], part[delta[i][0]], part[delta[i][1]])
            if sig not in sig_map:
                sig_map[sig] = len(sig_map)
            newpart[i] = sig_map[sig]
        if len(sig_map) == len(set(part)):
            part = newpart
            break
        part = newpart
    return len(set(part))


# ----------------------------- string sampling -------------------------------
def _gen_strings(u, randint, count, lmax, exclude=None):
    seen = set(exclude) if exclude else set()
    out = []
    tries = 0
    budget = count * 60 + 3000
    while len(out) < count and tries < budget:
        tries += 1
        L = randint(1, lmax)
        s = "".join("01"[randint(0, 1)] for _ in range(L))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ----------------------------- instance family --------------------------------
# (seed, k, n_train, Lmax_train, n_test, Lmax_test)
_SPECS = [
    (4101, 3, 70, 7, 130, 8),    # easy
    (4102, 4, 60, 7, 140, 12),   # trap: much longer held-out traces
    (4103, 3, 85, 6, 130, 7),    # easy
    (4104, 5, 65, 8, 150, 14),   # trap
    (4105, 4, 80, 7, 140, 8),    # moderate
    (4106, 6, 60, 8, 150, 16),   # trap: more states + long traces
    (4107, 3, 90, 6, 130, 7),    # easy
    (4108, 5, 55, 8, 140, 13),   # trap
    (4109, 4, 75, 7, 140, 9),    # moderate
    (4110, 7, 60, 9, 160, 18),   # hardest trap: most states, longest traces
]


def _build_one(dev_idx, seed, k, n_train, lmax_train, n_test, lmax_test):
    # Random target automata can induce a very skewed accept/reject distribution
    # over random strings (e.g. one absorbing reject state dominates), which makes
    # the majority-class baseline nearly unbeatable and hides real generalization
    # differences. Deterministically retry (varying the target + sampling seeds)
    # until BOTH the train and test label balance land in a workable range; fall
    # back to the first attempt if none qualifies (keeps this total & deterministic).
    fallback = None
    for attempt in range(80):
        delta, start, accept, _, _ = _build_target(seed * 7919 + 17 + attempt * 104729, k)
        u2, randint2 = _rng(seed * 1000003 + attempt * 97)
        train_strs = _gen_strings(u2, randint2, n_train, lmax_train)
        # NOTE: the held-out sample is drawn fresh from the SAME distribution as
        # training (typical train/test protocol) and is NOT forced disjoint from the
        # training strings -- for short lengths a literal repeat can legitimately
        # recur (a small combinatorial alphabet), while the bulk of the longer test
        # traces (see per-instance Lmax_test >> Lmax_train "trap" specs below)
        # necessarily lie outside anything a training string of bounded length could
        # equal, so genuine coincidental overlap cannot rescue a memorizer there.
        test_strs = _gen_strings(u2, randint2, n_test, lmax_test)
        if len(train_strs) < n_train or len(test_strs) < n_test:
            continue
        train = [{"s": s, "label": _run_dfa(delta, start, accept, s)} for s in train_strs]
        test = [(s, _run_dfa(delta, start, accept, s)) for s in test_strs]

        pos_train = sum(t["label"] for t in train) / len(train)
        pos_test = sum(y for _, y in test) / len(test)

        # a_triv is the accuracy of the TRAINING-majority rule (what "trivial.py"
        # actually computes: look only at the train labels, predict the more common
        # one for everything) -- NOT a test-label statistic, so it is a legitimate,
        # achievable-by-a-real-strategy anchor, not a hindsight oracle quantity.
        train_pos = sum(t["label"] for t in train)
        train_maj = 1 if train_pos * 2 >= len(train) else 0
        a_triv = sum(1 for _, y in test if y == train_maj) / len(test)
        a_oracle = sum(1 for s, y in test if _run_dfa(delta, start, accept, s) == y) / len(test)
        m = _minimize_count(delta, accept, k)

        result = {
            # public label is NOT the internal seed (avoids handing a candidate the
            # exact RNG seed it would need to try to regenerate the instance without
            # touching the training evidence at all).
            "name": "device_%02d" % dev_idx,
            "train": train,
            "test": test,
            "a_triv": a_triv,
            "a_oracle": a_oracle,
            "oracle_states": m,
        }
        if fallback is None:
            fallback = result
        if 0.35 <= pos_train <= 0.65 and 0.35 <= pos_test <= 0.65:
            return result
    return fallback


def _build_instances():
    return [_build_one(i, *spec) for i, spec in enumerate(_SPECS)]


# ----------------------------- candidate validation ----------------------------
def _validate_dfa(ans):
    if not isinstance(ans, dict):
        return None
    delta = ans.get("delta")
    start = ans.get("start")
    accept = ans.get("accept")
    if not isinstance(delta, list) or len(delta) == 0 or len(delta) > MAX_STATES:
        return None
    n = len(delta)
    clean_delta = []
    for row in delta:
        if not isinstance(row, list) or len(row) != 2:
            return None
        r = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                return None
            if v < 0 or v >= n:
                return None
            r.append(v)
        clean_delta.append(r)
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if start < 0 or start >= n:
        return None
    if not isinstance(accept, list):
        return None
    acc = set()
    for a in accept:
        if isinstance(a, bool) or not isinstance(a, int):
            return None
        if a < 0 or a >= n:
            return None
        acc.add(a)
    return clean_delta, start, acc


def _reachable_count(delta, start):
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for v in delta[x]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen)


def _dfa_acc(delta, start, accept, test):
    if not test:
        return 0.0
    correct = 0
    for s, y in test:
        st = start
        for ch in s:
            st = delta[st][0 if ch == "0" else 1]
        p = 1 if st in accept else 0
        if p == y:
            correct += 1
    return correct / len(test)


def score(inst, answer):
    v = _validate_dfa(answer)
    if v is None:
        return False, 0.0
    delta, start, accept = v
    a_cand = _dfa_acc(delta, start, accept, inst["test"])
    a_triv = inst["a_triv"]
    a_oracle = inst["a_oracle"]
    m = inst["oracle_states"]
    s = _reachable_count(delta, start)

    denom = max(a_oracle - a_triv, EPS_ACC)
    acc_r = (a_cand - a_triv) / denom
    if acc_r != acc_r:  # NaN guard
        acc_r = 0.0
    acc_r = max(0.0, min(1.0, acc_r))
    size_r = m / max(s, m)
    r = 0.1 + 0.75 * acc_r * (0.5 + 0.5 * size_r)
    if r != r or r in (float("inf"), float("-inf")):
        return False, 0.0
    r = max(0.0, min(1.0, r))
    return True, r


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {"name": inst["name"], "alphabet": ["0", "1"], "train": inst["train"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, r = score(inst, ans)
        except Exception:
            ok, r = False, 0.0
        vec.append(r if ok else 0.0)

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
