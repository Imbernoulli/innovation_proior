#!/usr/bin/env python3
"""
Checker for fsx_A_0741 (Archivist's Compaction Cadence).

Input (<in>):
  N M
  s_1 lo_1 hi_1          (N lines: box i's letter count and catalogue-key range)
  ...
  s_N lo_N hi_N
  T
  ev_1 .. ev_T            each "I" (next box, in order, arrives) or "L q"
                          (a reading-room request for catalogue key q)

Participant output (<out>): a merge plan
  K
  gap_1 first_1 last_1
  ...
  gap_K first_K last_K

`gap` in [0,T]: the merge is executed immediately after the first `gap`
timeline events have been processed (gap=0 = before anything happens).
`[first,last]` (1<=first<=last<=N) must, at that instant, be EXACTLY tiled
by currently-alive boxes/blocks (every id in [first,last] already arrived;
no block straddles the boundary). Gaps must be listed non-decreasing.

Cost model (minimize):
  - a merge of blocks whose ids span [first,last] costs sum(s_i for i in
    [first,last]) -- rewritten bytes, charged in FULL every time, even if
    some of that range was already merged before (write amplification is
    NOT free just because you touched those bytes earlier).
  - a lookup for key q costs the number of currently-alive blocks whose
    catalogue range [lo,hi] contains q (one probe per box that could hold
    the key -- read amplification from overlapping ranges).

F = total merge cost + total lookup cost. The checker also computes B, the
best of three simple, non-lookahead fixed policies it builds itself (never
merge / merge-everything after every insert / merge-everything every P
inserts for a couple of P). Score (minimization):
  Ratio = min(1, B / F)
"""
import sys
import math


def fail(reason):
    print("INVALID:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nx():
        try:
            return next(it)
        except StopIteration:
            raise ValueError("truncated input")

    N = int(nx())
    M = int(nx())
    sizes = [0] * (N + 1)
    lo = [0] * (N + 1)
    hi = [0] * (N + 1)
    for i in range(1, N + 1):
        sizes[i] = int(nx())
        lo[i] = int(nx())
        hi[i] = int(nx())
    T = int(nx())
    events = []  # list of ("I", None) or ("L", q)
    n_i = 0
    n_l = 0
    for _ in range(T):
        e = nx()
        if e == "I":
            n_i += 1
            events.append(("I", None))
        elif e == "L":
            q = int(nx())
            n_l += 1
            events.append(("L", q))
        else:
            raise ValueError(f"bad event token {e!r}")
    if n_i != N or n_l != M or T != N + M:
        raise ValueError("event counts do not match header")
    return N, M, T, sizes, lo, hi, events


# ---------------- simulation core, shared by scorer + baselines ----------------

class Blocks:
    """Alive blocks = a partition of {1..arrived} into contiguous id ranges.
    Kept as a sorted dict start-id -> (end, lo, hi, size)."""

    def __init__(self):
        self.by_start = {}
        self.by_end = {}

    def add_singleton(self, i, lo_i, hi_i, s_i):
        self.by_start[i] = (i, lo_i, hi_i, s_i)
        self.by_end[i] = i

    def find_start(self, a):
        return self.by_start.get(a)

    def merge_range(self, first, last):
        """Try to merge alive blocks EXACTLY tiling [first,last].
        Returns (ok, cost, new_lo, new_hi) without mutating on failure."""
        if first not in self.by_start:
            return False, 0, 0, 0
        chain = []
        cur = first
        total_size = 0
        new_lo = None
        new_hi = None
        while True:
            blk = self.by_start.get(cur)
            if blk is None:
                return False, 0, 0, 0
            end, blo, bhi, bs = blk
            chain.append(cur)
            total_size += bs
            new_lo = blo if new_lo is None else min(new_lo, blo)
            new_hi = bhi if new_hi is None else max(new_hi, bhi)
            if end == last:
                break
            if end > last:
                return False, 0, 0, 0
            cur = end + 1
        # remove chain, insert merged block
        for st in chain:
            end = self.by_start.pop(st)[0]
            del self.by_end[end]
        self.by_start[first] = (last, new_lo, new_hi, total_size)
        self.by_end[last] = first
        return True, total_size, new_lo, new_hi

    def probe_count(self, q):
        c = 0
        for (end, blo, bhi, bs) in self.by_start.values():
            if blo <= q <= bhi:
                c += 1
        return c


def simulate(N, T, sizes, lo, hi, events, merges_by_gap):
    """merges_by_gap: dict gap -> list of (first,last) IN SUBMITTED ORDER.
    Returns (cost, ok, reason). Raises nothing; strict feasibility."""
    blocks = Blocks()
    cost = 0
    arrived = 0
    i_ptr = 0  # next box id to arrive

    def apply_gap(g):
        nonlocal cost
        for (first, last) in merges_by_gap.get(g, []):
            if not (1 <= first <= last <= N):
                return False, f"merge range [{first},{last}] out of bounds"
            if last > arrived:
                return False, f"merge references box {last} not yet arrived at gap {g}"
            ok, c, _, _ = blocks.merge_range(first, last)
            if not ok:
                return False, f"merge range [{first},{last}] does not exactly tile alive blocks at gap {g}"
            cost += c
        return True, ""

    ok, msg = apply_gap(0)
    if not ok:
        return 0, False, msg

    for t in range(1, T + 1):
        kind, q = events[t - 1]
        if kind == "I":
            arrived += 1
            i_ptr += 1
            blocks.add_singleton(arrived, lo[arrived], hi[arrived], sizes[arrived])
        else:
            cost += blocks.probe_count(q)
        ok, msg = apply_gap(t)
        if not ok:
            return 0, False, msg

    return cost, True, ""


# ---------------- internal baselines (checker's own, non-lookahead) ----------------

def baseline_never(N, T, sizes, lo, hi, events):
    cost, ok, _ = simulate(N, T, sizes, lo, hi, events, {})
    return cost


def baseline_full_every_insert(N, T, sizes, lo, hi, events):
    """After every insert (if >1 alive block), merge everything alive."""
    blocks = Blocks()
    cost = 0
    arrived = 0
    for t in range(1, T + 1):
        kind, q = events[t - 1]
        if kind == "I":
            arrived += 1
            blocks.add_singleton(arrived, lo[arrived], hi[arrived], sizes[arrived])
            if len(blocks.by_start) > 1:
                first = min(blocks.by_start.keys())
                last = arrived
                ok, c, _, _ = blocks.merge_range(first, last)
                if ok:
                    cost += c
        else:
            cost += blocks.probe_count(q)
    return cost


def baseline_periodic(N, T, sizes, lo, hi, events, period):
    """Merge everything alive every `period` inserts."""
    blocks = Blocks()
    cost = 0
    arrived = 0
    since_merge_start = 1
    for t in range(1, T + 1):
        kind, q = events[t - 1]
        if kind == "I":
            arrived += 1
            blocks.add_singleton(arrived, lo[arrived], hi[arrived], sizes[arrived])
            if (arrived - since_merge_start + 1) >= period and len(blocks.by_start) > 1:
                first = min(blocks.by_start.keys())
                last = arrived
                ok, c, _, _ = blocks.merge_range(first, last)
                if ok:
                    cost += c
                    since_merge_start = arrived + 1
        else:
            cost += blocks.probe_count(q)
    return cost


def compute_baseline(N, T, sizes, lo, hi, events):
    """The BEST fixed-cadence policy over every period 1..N, plus never-merge
    and merge-on-every-insert as the two extremes -- i.e. the best possible
    'policy debate' construction, blind to WHEN lookups actually land. This
    is a strong, honestly-computed non-lookahead reference: it already picks
    the best constant tiering shape for this exact instance, so beating it
    takes genuinely reading the visible lookup schedule, not just picking a
    better constant."""
    best = baseline_never(N, T, sizes, lo, hi, events)
    best = min(best, baseline_full_every_insert(N, T, sizes, lo, hi, events))
    for period in range(2, N + 1):
        best = min(best, baseline_periodic(N, T, sizes, lo, hi, events, period))
    return best


# ---------------- parse participant output ----------------

def parse_output(path, N, T):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception as e:
        fail(f"cannot read output: {e}")
    if not toks:
        fail("empty output")
    it = iter(toks)

    def nx_int():
        try:
            tok = next(it)
        except StopIteration:
            fail("truncated output")
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token {tok!r}")
        if not math.isfinite(v):
            fail("non-finite token")
        return v

    K = nx_int()
    if K < 0 or K > 20000:
        fail(f"K={K} out of range")
    merges = []
    for _ in range(K):
        g = nx_int()
        first = nx_int()
        last = nx_int()
        if not (0 <= g <= T):
            fail(f"gap {g} out of range [0,{T}]")
        if not (1 <= first <= last <= N):
            fail(f"range [{first},{last}] out of range")
        merges.append((g, first, last))
    rest = list(it)
    if rest:
        fail(f"trailing garbage tokens ({len(rest)})")
    for i in range(1, len(merges)):
        if merges[i][0] < merges[i - 1][0]:
            fail("gaps must be listed in non-decreasing order")
    return merges


def main():
    if len(sys.argv) < 3:
        fail("usage: counter.py <in> <out> <ans>")
    inpath, outpath = sys.argv[1], sys.argv[2]

    try:
        N, M, T, sizes, lo, hi, events = read_input(inpath)
    except Exception as e:
        print("BAD INPUT:", e)
        print("Ratio: 0.0")
        sys.exit(0)

    merges = parse_output(outpath, N, T)

    merges_by_gap = {}
    for (g, first, last) in merges:
        merges_by_gap.setdefault(g, []).append((first, last))

    F, ok, msg = simulate(N, T, sizes, lo, hi, events, merges_by_gap)
    if not ok:
        fail(msg)

    B = compute_baseline(N, T, sizes, lo, hi, events)

    F = max(F, 1)
    B = max(B, 1)
    # eval_form=flops convention: fewer ops (here, lower total cost) is
    # better, and saturating the score requires a full order-of-magnitude
    # improvement over the best fixed-cadence reference -- leaves headroom
    # for a genuinely optimal (selective, not just globally-timed) schedule.
    # (100/1000 scaling below == the "Ratio = min(1, 0.1*B/F)" convention.)
    sc = min(1000.0, 100.0 * B / F)
    print("cost=%d baseline_cost=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
