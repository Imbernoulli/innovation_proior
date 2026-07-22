#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training corpus to stdout.

The Kessar Tablets (hidden-trs-induction).  Each testId fixes a DIFFERENT
hidden length-reducing string-rewriting system ("grammar") over the alphabet
{A,B,C}: rules L->R with |L| in [2,5], |R|==1, and a hidden total priority
order used to break ties when several rules' left-hand sides match the same
left-most position (this happens because left-hand sides of different rules
may overlap/nest as prefixes of one another).

STDOUT prints ONLY: a header "<N> <t>" then N lines "<word> <root>", where
word is a short (length <=9) attested string and root = its fully-reduced
normal form under the TRUE hidden grammar.  Coverage is EXHAUSTIVE for all
length-2 and length-3 windows (so the "short" rules are never ambiguous) and
a deterministic PARTIAL sample of length-4/length-5 windows -- exactly enough
that a careful, length-by-length bootstrap can identify the longer rules, but
a learner that never looks past length-3 windows cannot represent them at
all.  A handful of extra medium-length (6-9) words are included for flavour.

The hidden grammar, its priority order, and the held-out (length 24-36)
extrapolation set are regenerated identically -- and ONLY -- inside verify.py.
Nothing about the hidden law is printed here.
"""
import sys
import itertools
import random

ALPHABET = "ABC"

K2_RANGE = (3, 4)
K3_RANGE = (3, 4)
K4_RANGE = (2, 3)
K5_RANGE = (1, 2)
F4_SAMPLE = 0.55
F5_SAMPLE = 0.30
MEDIUM_N = 60
MEDIUM_LO, MEDIUM_HI = 6, 9

HIDDEN_SEED_BASE = 900001
HIDDEN_SEED_MULT = 7919
TRAIN_SEED_BASE = 31337
TRAIN_SEED_MULT = 104729


def all_strings(L):
    return [''.join(p) for p in itertools.product(ALPHABET, repeat=L)]


ALL2, ALL3, ALL4, ALL5 = all_strings(2), all_strings(3), all_strings(4), all_strings(5)


def build_hidden_grammar(t):
    """Hidden grammar for this tablet id: list of (priority, L, R).  Lives in
    gen.py AND verify.py, never printed."""
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
    """rules: iterable of (priority, L, R) -> dict L -> (priority, R), keeping
    the best (lowest-priority-number) rule per L."""
    idx = {}
    for pr, lhs, rhs in rules:
        cur = idx.get(lhs)
        if cur is None or pr < cur[0]:
            idx[lhs] = (pr, rhs)
    return idx


def reduce_string(s, idx, min_len=2, max_len=5):
    """Leftmost position, highest-priority (lowest number) matching rule
    fires; repeat to a fixed point.  Every rule strictly shortens the
    string, so this always terminates within len(s) firings."""
    steps = 0
    max_steps = len(s) + 50
    changed = True
    while changed and steps <= max_steps:
        changed = False
        i = 0
        n = len(s)
        while i < n:
            best = None
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


def gen_training_pairs(t, rules):
    rng = random.Random(TRAIN_SEED_BASE + t * TRAIN_SEED_MULT)
    idx = build_index(rules)
    pairs = []
    for pool, frac in ((ALL2, 1.0), (ALL3, 1.0), (ALL4, F4_SAMPLE), (ALL5, F5_SAMPLE)):
        k = max(1, int(round(len(pool) * frac)))
        k = min(k, len(pool))
        sample = rng.sample(pool, k)
        sample.sort()  # deterministic order regardless of sample's internal order
        for x in sample:
            pairs.append((x, reduce_string(x, idx)))
    for _ in range(MEDIUM_N):
        L = rng.randint(MEDIUM_LO, MEDIUM_HI)
        x = ''.join(rng.choice(ALPHABET) for _ in range(L))
        pairs.append((x, reduce_string(x, idx)))
    return pairs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rules = build_hidden_grammar(t)
    pairs = gen_training_pairs(t, rules)
    out = ["%d %d" % (len(pairs), t)]
    for w, r in pairs:
        out.append("%s %s" % (w, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
