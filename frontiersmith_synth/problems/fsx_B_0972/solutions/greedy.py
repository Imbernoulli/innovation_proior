# TIER: greedy
# The obvious first attempt: look ONLY at the batch of lines that just arrived
# this round, find the single phrase whose net token savings this round is
# largest, and coin it immediately if the codebook still has room. No memory
# of earlier rounds at all -- purely reactive to whatever looks good RIGHT NOW.
# This chases local/prefix frequency and can burn the whole lifetime cap on
# phrases that happen to spike once and never come back.
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
    # library: list of (id, pattern tuple); prefer the LONGEST match.
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
    lines = inst["lines"]
    cap = inst["cap"]
    overhead = inst["overhead"]
    library = [(m["id"], tuple(m["pattern"])) for m in inst["library"]]
    already = {p for _, p in library}

    new_macros = []
    if len(library) < cap:
        cand = best_candidate(lines, overhead, already)
        if cand is not None:
            new_macros.append({"pattern": list(cand)})

    lib_for_rewrite = library + [
        (len(library) + i, tuple(nm["pattern"])) for i, nm in enumerate(new_macros)
    ]
    rewrites = [encode_line(line, lib_for_rewrite) for line in lines]
    print(json.dumps({"new_macros": new_macros, "rewrites": rewrites}))


main()
