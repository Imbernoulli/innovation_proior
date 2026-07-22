#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1023 (cursor-biased-rope, format C, minimize).

Model (see statement.md for the full narrative):
  - Positions 1..N on the scroll. A fixed CENTRAL SPINDLE anchor sits at
    C0 = (N+1)//2 forever, free to use, never moves.
  - Up to F "finger" bookmarks (ids 1..F) start unset. A relocation event
    (t, i, q) moves bookmark i to position q, taking effect immediately
    before step t is serviced; cost = |old - q| + 2 where `old` is C0 if
    bookmark i has never been placed before ("fetched from the shelf").
    This is a LINEAR (rebuild-scale) cost.
  - Servicing step t (touching position p_t) costs the SMALLEST hop cost
    over the spindle and every currently-placed bookmark, where
    hop_cost(a, b) = bitlength(|a-b|) + 1. This is a LOGARITHMIC
    (distance-sublinear) cost -- the standard finger-search-tree property
    that access cost depends on distance to the nearest anchor, not on N.
  - Objective F = total relocation cost + total service cost (minimize).
"""
import sys

MAX_TOKENS = 20000
K_MAX = 5000


def read_instance(path):
    toks = open(path).read().split()
    N = int(toks[0]); M = int(toks[1]); F = int(toks[2])
    pts = [int(x) for x in toks[3:3 + M]]
    return N, M, F, pts


def hop_cost(a, b):
    return abs(a - b).bit_length() + 1


def reloc_cost(a, b):
    return abs(a - b) // 4 + 2


def parse_events(text, N, M, F):
    """Return (events_by_t, reason). events_by_t = list of length M+1, each a
    list of (i, q) pairs to apply before step t. None on failure."""
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        return None, "non-integer token (nan/inf/garbage)"
    ptr = 0
    K = vals[ptr]; ptr += 1
    if K < 0 or K > K_MAX:
        return None, f"K={K} out of range [0,{K_MAX}]"
    need = 3 * K
    if ptr + need != len(vals):
        return None, "token count does not match K (missing or trailing tokens)"
    events_by_t = [[] for _ in range(M + 1)]
    seen = set()
    for _ in range(K):
        t = vals[ptr]; i = vals[ptr + 1]; q = vals[ptr + 2]; ptr += 3
        if t < 1 or t > M:
            return None, f"event time {t} out of range [1,{M}]"
        if i < 1 or i > F:
            return None, f"finger id {i} out of range [1,{F}]"
        if q < 1 or q > N:
            return None, f"target position {q} out of range [1,{N}]"
        if (t, i) in seen:
            return None, f"duplicate relocation event (t={t}, i={i})"
        seen.add((t, i))
        events_by_t[t].append((i, q))
    return events_by_t, "ok"


def replay(N, M, F, pts, events_by_t):
    C0 = (N + 1) // 2
    finger_pos = [C0] * (F + 1)
    placed = [False] * (F + 1)
    total = 0
    for t in range(1, M + 1):
        for (i, q) in events_by_t[t]:
            total += reloc_cost(finger_pos[i], q)
            finger_pos[i] = q
            placed[i] = True
        p = pts[t - 1]
        best = hop_cost(C0, p)
        for i in range(1, F + 1):
            if placed[i]:
                c = hop_cost(finger_pos[i], p)
                if c < best:
                    best = c
        total += best
    return total


def baseline_cost(N, M, pts):
    C0 = (N + 1) // 2
    return sum(hop_cost(C0, p) for p in pts)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    inf, outf = sys.argv[1], sys.argv[2]
    N, M, F, pts = read_instance(inf)

    text = open(outf).read()
    events_by_t, reason = parse_events(text, N, M, F)
    if events_by_t is None:
        print(f"infeasible: {reason}")
        print("Ratio: 0.0")
        return 0

    Fcost = replay(N, M, F, pts, events_by_t)
    if not isinstance(Fcost, int) or Fcost < 0:
        print("non-finite/negative objective")
        print("Ratio: 0.0")
        return 0

    B = baseline_cost(N, M, pts)
    B = max(B, 1)

    sc = min(1000.0, 100.0 * B / max(1e-9, float(Fcost)))
    print(f"F={Fcost} baseline={B}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
