# TIER: strong
"""
The insight: process window lengths in ASCENDING order (2, then 3, then 4,
then 5).  At each length L, for every training pair (w, r) with |w| == L,
first check whether the rules ALREADY confirmed (from strictly shorter
lengths) already explain (w, r) under the same leftmost/highest-priority
procedure.  If they do, (w, r) needs no new rule -- it is either fully
explained by short rules, or explained by a longer rule that is permanently
masked by an overlapping short rule (and hence behaviourally irrelevant,
safe to omit).  If they do NOT, a genuine length-L rule must be responsible
(L -> r, since a single top-priority firing at position 0 is the only way a
length-L window can resolve to a single glyph that the shorter rules cannot
already produce) -- add it, with priority BETTER than every previously
confirmed (shorter) rule.

This is a composition argument, not a bigger dictionary: it recovers exactly
the rules that survive to matter under the true leftmost/priority semantics,
including the length-4/5 rules a naive length<=3 diff can never even
represent, and it composes correctly on long, never-jointly-observed strings
because each confirmed rule was validated against the SAME rewrite procedure
used at grading time.
"""
import sys


def build_index(rules):
    idx = {}
    for pr, lhs, rhs in rules:
        cur = idx.get(lhs)
        if cur is None or pr < cur[0]:
            idx[lhs] = (pr, rhs)
    return idx


def reduce_string(s, idx, min_len=2, max_len=5):
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


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0 AA A")
        return
    idx = 0
    n = int(data[idx]); idx += 1
    _t = int(data[idx]); idx += 1
    by_len = {2: [], 3: [], 4: [], 5: []}
    for _ in range(n):
        w = data[idx]; idx += 1
        r = data[idx]; idx += 1
        if len(w) in by_len:
            by_len[len(w)].append((w, r))

    known = []          # list of (priority, L, R); priority 0 = best confirmed so far

    for L in (2, 3, 4, 5):
        idxmap = build_index(known)
        found = []
        for w, r in by_len[L]:
            pred = reduce_string(w, idxmap)
            if pred != r and len(r) == 1:
                found.append((w, r))
        # this length's rules get better (lower) priority than everything
        # confirmed so far: shift all previous priorities up, insert new ones at 0.
        shift = len(found)
        known = [(pr + shift, lhs, rhs) for (pr, lhs, rhs) in known]
        for j, (w, r) in enumerate(found):
            known.append((j, w, r))

    # re-derive final priorities directly from `known` (already consistent)
    final_idx = build_index(known)
    lines = []
    for lhs, (pr, rhs) in final_idx.items():
        lines.append("%d %s %s" % (pr, lhs, rhs))
    if not lines:
        lines.append("0 AA A")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
