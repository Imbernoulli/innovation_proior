# TIER: strong
# Same capacity-K, deadline-ordered chunking per direction as the greedy
# recipe -- the insight is entirely in the SEQUENCING. Water cost only cares
# about a one-bit parity (repeat the previous lockage's direction or not),
# and for two colors the number of adjacent same-color pairs among the c0
# (majority) chunks is EXACTLY c0 minus how many non-empty majority "runs"
# you end up with. Using R = min(c0, c1+1) runs reaches the true floor
# max(0, c0-c1-1) -- and that floor does not care how big each run is or
# in what order, only that there are R of them. That slack is spent on
# urgency instead of forced into equal blocks: walk the majority queue and
# spend a "switch" to the minority queue the moment it is both allowed
# (there is already >=1 majority chunk in the current run) and either
# URGENT (the next minority chunk is more pressing) or MANDATORY (not
# enough majority chunks remain to fill every future run if we delay).
import sys


def build_chunks(boats, dirn, K):
    pool = sorted([b for b in boats if b[0] == dirn], key=lambda b: (b[2], b[1]))
    chunks = []
    for start in range(0, len(pool), K):
        group = pool[start:start + K]
        if not group:
            continue
        ids = [b[4] for b in group]
        max_a = max(b[1] for b in group)
        min_dl = min(b[2] for b in group)
        chunks.append((min_dl, max_a, ids))
    return chunks


def arrange(major, minor):
    """major/minor: chunk lists (min_dl, max_a, ids) each sorted by min_dl
    ascending, len(major) >= len(minor). Returns a sequence of (label,
    chunk) with label 0=major/1=minor achieving R=min(c0,c1+1) majority
    runs (the invariant-optimal same-label-adjacency floor) while
    scheduling switches by urgency."""
    c0, c1 = len(major), len(minor)
    if c0 == 0:
        return [(1, ch) for ch in minor]
    R = min(c0, c1 + 1)
    separators_needed = R - 1
    separators = minor[:separators_needed]
    leftover = minor[separators_needed:]

    seq = []
    mp = 0; sp = 0
    cur_run_len = 0
    while mp < c0 or sp < separators_needed:
        if mp >= c0:
            seq.append((1, separators[sp])); sp += 1
            continue
        if sp >= separators_needed:
            seq.append((0, major[mp])); mp += 1; cur_run_len += 1
            continue
        sep_remaining = separators_needed - sp
        mandatory = cur_run_len >= 1 and (c0 - mp) <= sep_remaining
        safe = cur_run_len >= 1 and (c0 - mp) >= sep_remaining
        urgent = separators[sp][0] < major[mp][0]
        if mandatory or (safe and urgent):
            seq.append((1, separators[sp])); sp += 1; cur_run_len = 0
        else:
            seq.append((0, major[mp])); mp += 1; cur_run_len += 1

    if leftover:
        lo = leftover[0]
        if seq and lo[0] < seq[0][1][0]:
            seq = [(1, lo)] + seq
        else:
            seq = seq + [(1, lo)]
    return seq


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); L = int(next(it))
    Wsame = int(next(it)); Wdiff = int(next(it)); s0 = int(next(it))
    boats = []
    for i in range(N):
        d = int(next(it)); a = int(next(it)); dd = int(next(it)); w = int(next(it))
        boats.append((d, a, dd, w, i + 1))

    chunks0 = build_chunks(boats, 0, K)
    chunks1 = build_chunks(boats, 1, K)

    if len(chunks0) >= len(chunks1):
        raw = arrange(chunks0, chunks1)
        seq = [(0 if lab == 0 else 1, ch) for (lab, ch) in raw]
    else:
        raw = arrange(chunks1, chunks0)
        seq = [(1 if lab == 0 else 0, ch) for (lab, ch) in raw]

    # free symmetry: reversing the whole sequence preserves the same-label
    # adjacency count, so use it to align the very first lockage with s0
    # when that's a free win.
    if seq and seq[0][0] == s0 and seq[-1][0] != s0:
        seq = list(reversed(seq))

    lockages = []
    t_prev = None
    for (dirn, (min_dl, max_a, ids)) in seq:
        t = max_a if t_prev is None else max(t_prev + L, max_a)
        lockages.append((t, dirn, ids))
        t_prev = t

    out = [str(len(lockages))]
    for (t, dirn, ids) in lockages:
        out.append(f"{t} {dirn} {len(ids)} " + " ".join(map(str, ids)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
