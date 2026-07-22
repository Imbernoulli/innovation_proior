#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for the pawn-shop shelf (offline generalized
caching with an explicit admission/bypass action) problem.

Replays the participant's per-visit actions against the checker's own ground-truth
shelf state (never trusting the participant's claim of hit/miss), validates every
action strictly, sums the total fetch (handling) cost F, and normalizes against the
checker's own trivial "always bypass" baseline B.

    minimization: sc = min(1000, 100*B/max(1e-9,F));  Ratio = sc/1000

Deterministic, O(N) amortized (each resident id is inserted/evicted O(1) times total),
no randomness, no wall-clock, exits 0 always with the LAST line 'Ratio: <float>'.
"""
import sys
import math


def die(reason):
    print("INFEASIBLE:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        try:
            return next(it)
        except StopIteration:
            die("truncated instance file")

    N = int(nxt()); K = int(nxt()); C = int(nxt())
    if N <= 0 or K <= 0 or C <= 0:
        die("bad header")
    sizes = [0] * (K + 1)
    for i in range(1, K + 1):
        sizes[i] = int(nxt())
        if sizes[i] <= 0:
            die("non-positive size")
    seq = [0] * N
    for i in range(N):
        v = int(nxt())
        if v < 1 or v > K:
            die("visit id out of range")
        seq[i] = v
    return N, K, C, sizes, seq


def is_finite_token(tok):
    try:
        x = float(tok)
    except ValueError:
        return False, None
    if math.isnan(x) or math.isinf(x):
        return False, None
    return True, x


def main():
    if len(sys.argv) < 3:
        die("usage: verify.py <in> <out> <ans>")
    inf, outf = sys.argv[1], sys.argv[2]
    N, K, C, sizes, seq = read_instance(inf)

    # checker's own trivial baseline: bypass every single visit -> every visit is a
    # (feasible, capacity-respecting-by-construction) miss.
    B = sum(sizes[v] for v in seq)
    if B <= 0:
        die("degenerate baseline")

    try:
        with open(outf, "r") as f:
            out_txt = f.read()
    except FileNotFoundError:
        die("missing output file")

    lines = out_txt.split("\n")
    # strip a single trailing empty line from the final newline, but do NOT silently
    # swallow genuinely blank action lines in the middle of the output.
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != N:
        die(f"expected exactly {N} action lines, got {len(lines)}")

    resident = {}   # id -> size, for ids currently on the shelf
    used = 0
    F = 0

    for i in range(N):
        oid = seq[i]
        s = sizes[oid]
        raw = lines[i]
        toks = raw.split()
        if not toks:
            die(f"empty action line at visit {i+1}")
        for t in toks:
            ok_fin, _ = is_finite_token(t) if t not in ("H", "B", "A") else (True, None)
            if not ok_fin:
                die(f"non-finite token at visit {i+1}")

        if oid in resident:
            # ground truth says this visit is a HIT: no cost, no state change.
            if toks[0] != "H" or len(toks) != 1:
                die(f"visit {i+1}: object {oid} is resident (a hit) but action was {raw!r}, expected 'H'")
            continue

        # ground truth: this visit is a MISS -> always pay the fetch cost.
        F += s

        tag = toks[0]
        if tag == "B":
            if len(toks) != 1:
                die(f"visit {i+1}: malformed bypass line {raw!r}")
            continue
        if tag != "A":
            die(f"visit {i+1}: object {oid} is a miss, action must be 'B' or 'A ...', got {raw!r}")
        if len(toks) < 2:
            die(f"visit {i+1}: malformed admit line {raw!r}")
        try:
            k = int(toks[1])
        except ValueError:
            die(f"visit {i+1}: admit eviction count is not an integer")
        if k < 0:
            die(f"visit {i+1}: negative eviction count")
        if len(toks) != 2 + k:
            die(f"visit {i+1}: admit line declares {k} evictions but has {len(toks)-2} id tokens")
        evict_ids = []
        seen = set()
        for j in range(k):
            try:
                eid = int(toks[2 + j])
            except ValueError:
                die(f"visit {i+1}: eviction id not an integer")
            if eid in seen:
                die(f"visit {i+1}: duplicate eviction id {eid}")
            seen.add(eid)
            if eid not in resident:
                die(f"visit {i+1}: eviction id {eid} is not currently on the shelf")
            evict_ids.append(eid)
        freed = sum(resident[e] for e in evict_ids)
        if used - freed + s > C:
            die(f"visit {i+1}: admitting object {oid} (size {s}) after evicting {evict_ids} "
                f"still exceeds capacity ({used-freed+s} > {C})")
        if s > C:
            die(f"visit {i+1}: object {oid} (size {s}) cannot ever fit in capacity {C}")
        for e in evict_ids:
            used -= resident.pop(e)
        resident[oid] = s
        used += s

    if used != sum(resident.values()):
        die("internal accounting mismatch")  # defensive; should be unreachable

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = max(0.0, sc) / 1000.0
    print(f"F={F} B={B}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
