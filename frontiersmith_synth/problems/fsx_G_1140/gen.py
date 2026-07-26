#!/usr/bin/env python3
"""
gen.py <testId>  -- prints ONE training instance for the tropical-weight-automaton
recovery task to stdout.

Hidden law (NEVER printed): a 2-state min-plus (tropical) weighted automaton over
alphabet {a,b}. State 0 ("slow") is the start state; reading 'b' always stays in
state 0 at cost p_b; reading 'a' in state 0 costs p_a (stay) OR q (switch to state
1, "fast", a one-time investment). Once in state 1, 'a' costs r_a and 'b' costs
r_b per symbol (both cheaper than the slow rates). Both states are accepting with
final weight 0. Because switching is always at least as good done at the FIRST
'a' (never later, since r_a<p_a and r_b<p_b), the true cost of a string is a
genuine MIN of two route totals -- a piecewise / regime-switching function of the
string, not a single additive (per-symbol-count) formula.

Only testId seeds everything (deterministic). STDOUT prints TRAIN rows only
(string, cost) -- never the seed, the weights, or the law.
"""
import sys, random

ALPHA = "ab"


def hidden_params(t):
    """Deterministic hidden automaton weights + train-length scale, from testId only."""
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
    """Min-plus DP over the hidden 2-state automaton. Both states accepting, final=0."""
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
    out = [(s, true_cost(s, p_a, p_b, q, r_a, r_b)) for s in rows]
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    rows = gen_train_rows(t)
    out = []
    out.append("%d %d" % (t, len(rows)))
    out.append("alphabet a b")
    for s, c in rows:
        # costs are exact integers under the hidden law (integer weights)
        out.append("%s %d" % (s, int(round(c))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
