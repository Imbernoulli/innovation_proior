#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the town-meeting bundle-referendum task.

Instance (from <in>):
    line 1: P V K
    then for i = 0..P-1, three lines:
      w_i s_i o_i ns_i no_i
      ns_i supporter voter indices
      no_i opposer voter indices
  Every voter not listed for project i is NEUTRAL on i (valuation 0).

Participant artifact (from <out>):
    line 1: P                     (must echo the instance's P)
    line 2: P integers b_1..b_P, each in [1,K] -- project i (0-indexed, in
            the SAME order the instance listed projects) is assigned to
            bundle b_{i+1}.

Bundle-pass rule.  For bundle id k, and for every voter v, sum the voter's
valuation over every project assigned to k:
    val_i(v) = +s_i  if v in supporters_i
             = -o_i  if v in opposers_i
             =  0    otherwise
    total_k(v) = sum_{i: assign[i]=k} val_i(v)
Bundle k PASSES iff strictly more than half the voters have total_k(v) > 0
(voters with total_k(v) <= 0, including exact cancellations, do not count).
Empty bundles trivially do not pass (contribute nothing).

Objective (maximise): the sum of welfare weights w_i over every project i
whose assigned bundle passes.

Scoring.  Internal baseline B = the grader's own trivial construction:
project 0 (the generator's guaranteed-safe consensus project) sits alone in
bundle 1; every other project is lumped together into bundle 2 (bundles
3..K unused).  B = w_0 + (sum of the other weights, IF that lump passes,
else 0); B is always >= w_0 > 0 by construction.
    sc    = min(1000, 100 * F / max(1e-9, B))
    Ratio = max(0, sc) / 1000
so reproducing the baseline construction scores ~0.1, and an ~10x-better
partition would cap at 1.0.  Any feasibility violation prints Ratio: 0.0.
"""
import sys, math


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as fh:
        toks = fh.read().split()
    it = iter(toks)
    P = int(next(it)); V = int(next(it)); K = int(next(it))
    projects = []
    for _ in range(P):
        w = int(next(it)); s = int(next(it)); o = int(next(it))
        ns = int(next(it)); no = int(next(it))
        sup = [int(next(it)) for _ in range(ns)]
        opp = [int(next(it)) for _ in range(no)]
        projects.append((w, s, o, sup, opp))
    return P, V, K, projects


def bundle_total_score(members, projects, V):
    """Exact per-voter sum -> yes-count -> pass/fail, for a set of project indices."""
    sums = {}
    for i in members:
        w, s, o, sup, opp = projects[i]
        for v in sup:
            sums[v] = sums.get(v, 0) + s
        for v in opp:
            sums[v] = sums.get(v, 0) - o
    yes = sum(1 for val in sums.values() if val > 0)
    return yes * 2 > V


def score_partition(assign, P, V, K, projects):
    bundles = {}
    for i in range(P):
        bundles.setdefault(assign[i], []).append(i)
    total = 0
    for k, members in bundles.items():
        if bundle_total_score(members, projects, V):
            total += sum(projects[i][0] for i in members)
    return total


def baseline_value(P, V, K, projects):
    assign = [2] * P
    assign[0] = 1
    return score_partition(assign, P, V, K, projects)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    P, V, K, projects = read_instance(sys.argv[1])

    try:
        with open(sys.argv[2], "rb") as fh:
            raw = fh.read(1 << 20)
    except Exception:
        fail("cannot read output")
    text = raw.decode("utf-8", "replace")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        fail("need 2 non-blank lines (P line + assignment line)")

    try:
        p_echo = int(lines[0].split()[0])
    except Exception:
        fail("first line must be an integer P")
    if p_echo != P:
        fail("P mismatch: expected %d got %d" % (P, p_echo))

    toks = lines[1].split()
    if len(toks) != P:
        fail("assignment line must have exactly %d tokens, got %d" % (P, len(toks)))

    assign = []
    for tok in toks:
        try:
            b = int(tok)
        except Exception:
            fail("non-integer bundle id %r" % tok)
        if not math.isfinite(b):
            fail("non-finite bundle id")
        if b < 1 or b > K:
            fail("bundle id %d out of range [1,%d]" % (b, K))
        assign.append(b)

    if len(lines) > 2:
        # allow a trailing blank/garbage-free extra line only if it is empty;
        # any further non-blank token stream is a formatting violation.
        for extra in lines[2:]:
            if extra.split():
                fail("unexpected extra output after the assignment line")

    F = score_partition(assign, P, V, K, projects)
    B = baseline_value(P, V, K, projects)
    if B <= 0:
        B = 1e-6

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    if sc < 0.0:
        sc = 0.0
    print("F=%d B=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
