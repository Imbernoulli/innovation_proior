# TIER: strong
# Insight: this is voter-SET algebra, not weight balancing.  A bundle passes
# or fails according to the exact union of its members' supporter/opposer
# sets against the V voters -- so we compute, for every candidate bundle,
# the real per-voter sum instead of any popularity proxy.
#
#   1. Test every project alone; split into solo-pass and solo-fail.
#   2. For every solo-fail project p, search over every OTHER unused project
#      q and exactly recompute whether {p, q} passes (real per-voter sums,
#      no approximation).  This is a set-cover step over vote margins: q
#      "covers" p's shortfall iff the union of q's supporters with p's own
#      supporters crosses the majority threshold.  Among all q that flip p,
#      keep the one with the largest total welfare w_p + w_q.
#   3. Solve the resulting matching greedily by descending gain (an
#      unmatched flip is worth 0, so higher combined weight goes first),
#      capped by the K-1 bundles left after reserving one bundle for the
#      safe solo-passers.
#   4. Every solo-pass project not consumed by a flip pair is bundled
#      together (safe: no opposers among them, so the union can only add
#      yes votes, never lose any).
#   5. Anything left over (poison, unmatched solo-failers) is dumped into
#      one final bundle; it may fail, but there is nothing better to do
#      with it and it must not be allowed to contaminate a bundle that
#      would otherwise have passed.
import sys


def read():
    toks = sys.stdin.read().split()
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


def passes(members, projects, V):
    sums = {}
    for i in members:
        w, s, o, sup, opp = projects[i]
        for v in sup:
            sums[v] = sums.get(v, 0) + s
        for v in opp:
            sums[v] = sums.get(v, 0) - o
    yes = sum(1 for val in sums.values() if val > 0)
    return yes * 2 > V


def main():
    P, V, K, projects = read()
    weights = [pr[0] for pr in projects]

    solo_pass = [i for i in range(P) if passes([i], projects, V)]
    solo_fail = [i for i in range(P) if i not in solo_pass]

    # step 2: greedy-by-value matching.  Process solo-fail projects in
    # DESCENDING welfare weight so the highest-value polarizer gets first
    # pick of an available rescuer; re-search among currently UNUSED
    # projects each time (small "sweetener" projects are themselves
    # solo-fail and interchangeable rescuers, so a static one-shot
    # best-partner-per-project search can starve a high-value polarizer of
    # the partner a lower-value one already grabbed -- dynamic re-search
    # avoids that).
    assign = [0] * P
    used = [False] * P
    bundle_id = 1
    max_pairs = max(0, K - 1)  # keep >=1 bundle free for solo-passers/dump
    selected = []
    for p in sorted(solo_fail, key=lambda i: -weights[i]):
        if used[p] or len(selected) >= max_pairs:
            continue
        best_q, best_gain = None, -1
        for q in range(P):
            if q == p or used[q]:
                continue
            if passes([p, q], projects, V):
                gain = weights[p] + weights[q]
                if gain > best_gain:
                    best_gain, best_q = gain, q
        if best_q is not None:
            selected.append((p, best_q))
            used[p] = used[best_q] = True

    for p, q in selected:
        if bundle_id > K:
            break
        assign[p] = bundle_id
        assign[q] = bundle_id
        bundle_id += 1

    remaining_bundles = K - bundle_id + 1

    leftover_solo_pass = [i for i in solo_pass if not used[i]]
    if leftover_solo_pass:
        target = bundle_id if remaining_bundles >= 1 else K
        for i in leftover_solo_pass:
            assign[i] = target
            used[i] = True
        if remaining_bundles >= 1:
            bundle_id += 1
            remaining_bundles -= 1

    rest = [i for i in range(P) if not used[i]]
    if rest:
        dump = bundle_id if remaining_bundles >= 1 else K
        for i in rest:
            assign[i] = dump
            used[i] = True

    # safety net: anything unassigned (shouldn't happen) goes to bundle K
    for i in range(P):
        if assign[i] == 0:
            assign[i] = K

    print(P)
    print(" ".join(map(str, assign)))


if __name__ == "__main__":
    main()
