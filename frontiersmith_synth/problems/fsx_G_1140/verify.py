#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the tropical (min-plus) weighted-automaton recovery task.

The solver submits ONE small min-plus weighted automaton over alphabet {a,b}
(states, per-symbol transition weights -- possibly several transitions per
(state,symbol), nondeterministic-choice min-plus semantics -- a start state, and
a set of accepting states with final weights). We:

  1. Regenerate the hidden law + TRAIN rows from the testId in <in> (byte-for-byte
     identical procedure to gen.py) -- never trust <in>'s body for scoring math.
  2. Parse & strictly validate the submitted automaton (bounded sizes, finite
     weights, well-formed transitions, >=1 accepting state). Any violation ->
     Ratio 0.0.
  3. TRAIN-FIT GATE: evaluate the submitted automaton (its own min-plus DP) on
     every TRAIN string; if its mean relative error vs the true train costs
     exceeds a generous tolerance, the automaton was never fit to the data it was
     given -> Ratio 0.0.
  4. Regenerate a HELD-OUT EXTRAPOLATION set -- strings substantially LONGER than
     any train string, plus a few pure-'b' long strings -- with the SAME hidden
     automaton (small held-out sensor-style noise). Score = mean relative error
     of the submission on this set, compared against an internal baseline (a
     flat per-symbol-blind rate predictor fit from the same train data):

        O        = mean_i min(5, |predicted_i - true_i| / max(1, true_i))
        B        = same metric for the baseline rate*length predictor
        Ratio    = clip(0.9 - 0.8 * (O / B), 0, 1)

     A submission that matches the baseline scores ~0.1; O=0 (perfect, never
     achieved because of held-out noise) would cap at 0.9, leaving headroom.
"""
import sys, math, random

ALPHA_IDX = {"a": 0, "b": 1}
MAX_STATES = 8
MAX_TRANS = 64
MAX_WEIGHT = 1.0e5
MAX_OUT_BYTES = 200_000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------------- hidden law (identical to gen.py) ----------------
def hidden_params(t):
    rng = random.Random(500000 + t * 918273 + 17)
    p_a = rng.randint(16, 26)
    p_b = rng.randint(16, 26)
    tries = 0
    while abs(p_a - p_b) < 5 and tries < 50:
        p_b = rng.randint(16, 26)
        tries += 1
    gap_a = rng.randint(7, 13)
    gap_b = rng.randint(7, 13)
    r_a = max(2, p_a - gap_a)
    r_b = max(2, p_b - gap_b)
    Ltrain_max = 14 + 2 * (t - 1)
    target = max(4, round(0.6 * Ltrain_max))
    target = max(3, target + rng.randint(-1, 1))
    q = r_a + (p_a - r_a) * (target - 1)
    q += rng.randint(-2, 2)
    q = max(q, r_a + 1)
    return p_a, p_b, q, r_a, r_b, Ltrain_max


def true_cost(w, p_a, p_b, q, r_a, r_b):
    INF = float("inf")
    d0, d1 = 0.0, INF
    for ch in w:
        if ch == "a":
            n0 = d0 + p_a
            n1 = min(d0 + q, d1 + r_a)
        else:
            n0 = d0 + p_b
            n1 = d1 + r_b
        d0, d1 = n0, n1
    return min(d0, d1)


def gen_train_rows(t):
    p_a, p_b, q, r_a, r_b, Ltm = hidden_params(t)
    rng = random.Random(t * 7331 + 11)
    Ltrain_min = 3
    rows = []
    for _ in range(50):
        L = rng.randint(Ltrain_min, Ltm)
        s = "".join("a" if rng.random() < 0.5 else "b" for _ in range(L))
        rows.append(s)
    for i in range(10):
        L = max(2, round(2 + i * (Ltm - 2) / 9.0))
        rows.append("a" * L)
    for i in range(6):
        L = max(2, round(2 + i * (Ltm - 2) / 5.0))
        rows.append("b" * L)
    return [(s, true_cost(s, p_a, p_b, q, r_a, r_b)) for s in rows]


def gen_extrap_rows(t, params, noise_sd=0.08):
    p_a, p_b, q, r_a, r_b, Ltm = params
    rng = random.Random(t * 99991 + 31337)
    rng_noise = random.Random(t * 424242 + 5)
    Lmin, Lmax = Ltm + 20, Ltm + 60
    rows = []
    for _ in range(50):
        L = rng.randint(Lmin, Lmax)
        rows.append("".join("a" if rng.random() < 0.5 else "b" for _ in range(L)))
    for _ in range(8):
        L = rng.randint(Lmin, Lmax)
        rows.append("b" * L)
    out = []
    for s in rows:
        c = true_cost(s, p_a, p_b, q, r_a, r_b)
        c = c * (1.0 + rng_noise.gauss(0.0, noise_sd))
        out.append((s, max(0.0, c)))
    return out


# ---------------- parse + validate the submitted automaton ----------------
def parse_automaton(path):
    try:
        with open(path, "rb") as fh:
            blob = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(blob) > MAX_OUT_BYTES:
        fail("output too large")
    try:
        text = blob.decode("ascii")
    except Exception:
        fail("non-ascii output")
    toks = text.split()
    it = iter(toks)

    def next_int():
        return int(next(it))

    def next_float():
        v = float(next(it))
        if not math.isfinite(v):
            raise ValueError("non-finite")
        return v

    try:
        S = next_int()
        if not (1 <= S <= MAX_STATES):
            fail("state count out of range")
        T = next_int()
        if not (0 <= T <= MAX_TRANS):
            fail("transition count out of range")
        trans = [[[], []] for _ in range(S)]  # trans[i][0/1 for a/b] = [(j,w),...]
        for _ in range(T):
            i = next_int()
            c = next(it)
            j = next_int()
            w = next_float()
            if not (0 <= i < S and 0 <= j < S):
                fail("transition state index out of range")
            if c not in ALPHA_IDX:
                fail("transition symbol not in {a,b}")
            if abs(w) > MAX_WEIGHT:
                fail("transition weight out of range")
            trans[i][ALPHA_IDX[c]].append((j, w))
        start = next_int()
        if not (0 <= start < S):
            fail("start state out of range")
        K = next_int()
        if not (1 <= K <= S):
            fail("need 1..S accepting states")
        finals = {}
        for _ in range(K):
            j = next_int()
            f = next_float()
            if not (0 <= j < S):
                fail("final state index out of range")
            if abs(f) > MAX_WEIGHT:
                fail("final weight out of range")
            if j not in finals or f < finals[j]:
                finals[j] = f
    except StopIteration:
        fail("truncated output")
    except ValueError:
        fail("non-numeric / non-finite token")
    except SystemExit:
        raise
    except Exception:
        fail("malformed output")
    return S, trans, start, finals


def eval_cost(S, trans, start, finals, w):
    INF = float("inf")
    d = [INF] * S
    d[start] = 0.0
    for ch in w:
        idx = ALPHA_IDX[ch]
        nd = [INF] * S
        for i in range(S):
            di = d[i]
            if di == INF:
                continue
            for (j, wt) in trans[i][idx]:
                v = di + wt
                if v < nd[j]:
                    nd[j] = v
        d = nd
        if all(x == INF for x in d):
            return INF
    best = INF
    for j, f in finals.items():
        if d[j] < INF:
            v = d[j] + f
            if v < best:
                best = v
    return best


def rel_err(pred, true, cap=5.0):
    if pred == float("inf") or not math.isfinite(pred):
        return cap
    return min(cap, abs(pred - true) / max(1.0, true))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            first = fh.readline().split()
        t = int(first[0])
    except Exception:
        fail("bad instance header")

    params = hidden_params(t)
    train = gen_train_rows(t)
    extrap = gen_extrap_rows(t, params)

    S, trans, start, finals = parse_automaton(outf)

    # ---- TRAIN-FIT feasibility gate ----
    train_errs = [rel_err(eval_cost(S, trans, start, finals, s), c) for s, c in train]
    train_mre = sum(train_errs) / len(train_errs)
    TRAIN_TOL = 0.35
    if train_mre > TRAIN_TOL:
        fail("does not fit training data (train MRE=%.4f > %.2f)" % (train_mre, TRAIN_TOL))

    # ---- baseline: flat rate*length predictor fit from TRAIN ----
    total_c = sum(c for _, c in train)
    total_l = sum(len(s) for s, _ in train)
    rate = total_c / max(1.0, total_l)

    O = sum(rel_err(eval_cost(S, trans, start, finals, s), c) for s, c in extrap) / len(extrap)
    B = sum(rel_err(rate * len(s), c) for s, c in extrap) / len(extrap)

    ratio = 0.9 - 0.8 * (O / max(1e-9, B))
    ratio = max(0.0, min(1.0, ratio))
    print("train_MRE=%.4f held_out_MRE=%.4f baseline_MRE=%.4f Ratio: %.6f" % (train_mre, O, B, ratio))


if __name__ == "__main__":
    main()
