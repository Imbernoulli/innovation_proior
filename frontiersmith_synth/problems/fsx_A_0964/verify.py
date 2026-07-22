#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for The Kessar Tablets (hidden-trs-induction).

- Reads the tablet id `t` from <in>'s header, then regenerates the hidden
  grammar (list of (priority, L, R), |L| in [2,5], |R|==1) and a HELD-OUT set
  of 250 long (length 24-36) inscriptions -- a regime never shown in
  training -- entirely from `t`.  The hidden grammar lives ONLY here (and,
  identically, in gen.py -- never printed to the solver).
- Parses the participant's submitted rule set: lines "priority L R".
  Feasibility: exactly 3 tokens/line, priority a finite number, L in
  {A,B,C}^[2,5], R in {A,B,C}, at most 80 rules, non-empty submission. ANY
  violation -> Ratio 0.
- Reduces each held-out word twice -- once under the TRUE grammar, once under
  the SUBMITTED one -- via the identical leftmost / highest-priority-wins
  procedure, and scores the mean normalized-Levenshtein similarity of the
  roots against the mean similarity achieved by the identity baseline
  (predict the root = the raw word, i.e. submit no rules).
      Ratio = min(1000, 100 * F / B) / 1000
  Reproducing the SHORT training pairs is not checked at all -- only
  extrapolation to long inscriptions is graded, which is where rules whose
  left-hand side is 4-5 glyphs long (almost invisible in short words) start
  to dominate.
"""
import sys
import itertools
import random
import math

ALPHABET = "ABC"
ALPHASET = set(ALPHABET)

K2_RANGE = (3, 4)
K3_RANGE = (3, 4)
K4_RANGE = (2, 3)
K5_RANGE = (1, 2)

HIDDEN_SEED_BASE = 900001
HIDDEN_SEED_MULT = 7919
HELDOUT_SEED_BASE = 555111
HELDOUT_SEED_MULT = 15485863
HELD_N = 250
HELD_LO, HELD_HI = 24, 36

MAX_RULES = 80
MAX_OUT_BYTES = 100000
MIN_LHS, MAX_LHS = 2, 5


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def all_strings(L):
    return [''.join(p) for p in itertools.product(ALPHABET, repeat=L)]


ALL2, ALL3, ALL4, ALL5 = all_strings(2), all_strings(3), all_strings(4), all_strings(5)


# ---------- hidden grammar (identical to gen.py) ----------
def build_hidden_grammar(t):
    rng = random.Random(HIDDEN_SEED_BASE + t * HIDDEN_SEED_MULT)
    rules = []

    def add(pool, rr):
        cand = list(pool)
        rng.shuffle(cand)
        n = rng.randint(*rr)
        for lhs in cand[:n]:
            rules.append([lhs, rng.choice(ALPHABET)])

    add(ALL2, K2_RANGE)
    add(ALL3, K3_RANGE)
    add(ALL4, K4_RANGE)
    add(ALL5, K5_RANGE)
    rng.shuffle(rules)
    return [(i, lhs, rhs) for i, (lhs, rhs) in enumerate(rules)]


def build_index(rules):
    idx = {}
    for pr, lhs, rhs in rules:
        cur = idx.get(lhs)
        if cur is None or pr < cur[0]:
            idx[lhs] = (pr, rhs)
    return idx


def reduce_string(s, idx, min_len=MIN_LHS, max_len=MAX_LHS):
    steps = 0
    max_steps = len(s) + 50
    changed = True
    while changed and steps <= max_steps:
        changed = False
        i = 0
        n = len(s)
        while i < n:
            best = None
            best_len = 0
            for L in range(max_len, min_len - 1, -1):
                if i + L > n:
                    continue
                e = idx.get(s[i:i + L])
                if e is not None and (best is None or e[0] < best[0]):
                    best = e
                    best_len = L
            if best is not None:
                pr, rhs = best
                s = s[:i] + rhs + s[i + best_len:]
                n = len(s)
                changed = True
                steps += 1
                i = 0
            else:
                i += 1
    return s


def gen_heldout(t, idx):
    rng = random.Random(HELDOUT_SEED_BASE + t * HELDOUT_SEED_MULT)
    out = []
    for _ in range(HELD_N):
        L = rng.randint(HELD_LO, HELD_HI)
        x = ''.join(rng.choice(ALPHABET) for _ in range(L))
        out.append((x, reduce_string(x, idx)))
    return out


def lev(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def sim(a, b):
    d = lev(a, b)
    return 1.0 - d / max(len(a), len(b), 1)


# ---------- parse the submitted grammar ----------
def parse_submission(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty submission")
    if len(lines) > MAX_RULES:
        fail("too many rules (%d > %d)" % (len(lines), MAX_RULES))
    rules = []
    for li, ln in enumerate(lines):
        toks = ln.split()
        if len(toks) != 3:
            fail("line %d: expected 3 tokens, got %d" % (li + 1, len(toks)))
        ptok, lhs, rhs = toks
        try:
            pr = float(ptok)
        except ValueError:
            fail("line %d: priority not a number" % (li + 1))
        if not math.isfinite(pr):
            fail("line %d: non-finite priority" % (li + 1))
        if not (MIN_LHS <= len(lhs) <= MAX_LHS):
            fail("line %d: L length %d out of [%d,%d]" % (li + 1, len(lhs), MIN_LHS, MAX_LHS))
        if any(c not in ALPHASET for c in lhs):
            fail("line %d: L uses out-of-alphabet character" % (li + 1))
        if len(rhs) != 1 or rhs not in ALPHASET:
            fail("line %d: R must be a single glyph in {A,B,C}" % (li + 1))
        # tie-break ties by original line order (stable): encode as (pr, li)
        rules.append(((pr, li), lhs, rhs))
    return rules


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 1000000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    rules = parse_submission(text)
    sub_idx = build_index(rules)

    hidden = build_hidden_grammar(t)
    hidden_idx = build_index(hidden)
    heldout = gen_heldout(t, hidden_idx)

    tot_f = 0.0
    tot_b = 0.0
    for w, true_root in heldout:
        pred_root = reduce_string(w, sub_idx)
        tot_f += sim(pred_root, true_root)
        tot_b += sim(w, true_root)
    F = tot_f / len(heldout)
    B = tot_b / len(heldout)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f n_rules=%d  Ratio: %.6f" % (F, B, len(rules), sc / 1000.0))


if __name__ == "__main__":
    main()
