# TIER: strong
# The insight: raw within-batch frequency is a bad proxy for which phrase is
# worth an irrevocable shorthand slot. A phrase that spikes hard in exactly
# one round and never returns is weak evidence of real session-wide structure
# -- coining it can burn the (small, permanent) codebook before the true
# latent template of the *whole* session is identifiable, since that template
# may only be weakly present in early rounds. So:
#   - never coin anything in round 0 (no possible multi-round evidence yet:
#     reserve the budget instead of chasing whatever looks big on day one);
#   - from round 1 on, first check whether any candidate phrase has now
#     recurred in >= 2 DISTINCT rounds (real evidence of persistent
#     structure, not a one-off) and, if so, spend one slot on the best such
#     phrase by realized net-benefit so far;
#   - only THEN, if the codebook still has room, also grab this round's best
#     fresh local spike (same rule "greedy" uses) so a genuinely valuable
#     one-off is not left on the table once the reserved slot is safely spent.
# This is the exploration/commitment trade-off made concrete: spend one
# reserved slot on cross-round evidence before spending the rest on frequency.
import sys, json
from collections import Counter

LMIN, LMAX = 2, 12


def window_counts(lines, L):
    cnt = Counter()
    for line in lines:
        n = len(line)
        for i in range(n - L + 1):
            cnt[tuple(line[i:i + L])] += 1
    return cnt


def is_sub(a, b):
    la, lb = len(a), len(b)
    if la > lb:
        return False
    return any(b[i:i + la] == a for i in range(lb - la + 1))


def dominated(p, already):
    for c in already:
        if is_sub(p, c) or is_sub(c, p):
            return True
    return False


def best_candidate(lines, overhead, already):
    best = None
    for L in range(LMIN, LMAX + 1):
        cnt = window_counts(lines, L)
        for p, c in cnt.items():
            if p in already or dominated(p, already):
                continue
            benefit = c * (L - 1) - (L + overhead)
            if benefit > 0 and (best is None or benefit > best[1]):
                best = (p, benefit)
    return best[0] if best else None


def encode_line(line, library):
    lib_sorted = sorted(library, key=lambda kp: (-len(kp[1]), kp[0]))
    out = []
    i, n = 0, len(line)
    while i < n:
        matched = False
        for mid, pat in lib_sorted:
            Lp = len(pat)
            if i + Lp <= n and tuple(line[i:i + Lp]) == pat:
                out.append(f"${mid}")
                i += Lp
                matched = True
                break
        if not matched:
            out.append(line[i])
            i += 1
    return out


def main():
    inst = json.load(sys.stdin)
    r = inst["round"]
    lines = inst["lines"]
    history = inst["history"]
    cap = inst["cap"]
    overhead = inst["overhead"]
    library = [(m["id"], tuple(m["pattern"])) for m in inst["library"]]
    already = {p for _, p in library}

    new_patterns = []
    budget = cap - len(library)

    if budget > 0 and r > 0:
        all_rounds = history + [lines]
        rounds_seen = {}
        cum_count = {}
        for L in range(LMIN, LMAX + 1):
            for ridx, rlines in enumerate(all_rounds):
                for p, c in window_counts(rlines, L).items():
                    rounds_seen.setdefault(p, set()).add(ridx)
                    cum_count[p] = cum_count.get(p, 0) + c

        recurring = [
            p for p in rounds_seen
            if len(rounds_seen[p]) >= 2 and p not in already and not dominated(p, already)
        ]

        def cum_benefit(p):
            Lp = len(p)
            return cum_count[p] * (Lp - 1) - (Lp + overhead)

        recurring_pos = [p for p in recurring if cum_benefit(p) > 0]
        if recurring_pos:
            best = max(recurring_pos, key=cum_benefit)
            new_patterns.append(list(best))
            already = already | {best}
            budget -= 1

        if budget > 0:
            cand = best_candidate(lines, overhead, already)
            if cand is not None:
                new_patterns.append(list(cand))
                budget -= 1

    new_macros = [{"pattern": p} for p in new_patterns]
    lib_for_rewrite = library + [
        (len(library) + i, tuple(p)) for i, p in enumerate(new_patterns)
    ]
    rewrites = [encode_line(line, lib_for_rewrite) for line in lines]
    print(json.dumps({"new_macros": new_macros, "rewrites": rewrites}))


main()
