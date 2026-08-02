#!/usr/bin/env python3
"""gen.py <testId> -- one instance of the cell-layout / interleave-distance /
decoder-cost problem (family: error-correcting-layout).

Deterministic: everything is derived from testId only (no wall clock, no OS entropy).

Instance (stdout):
    N M LMAX
    then M lines: w t cost      (code catalog, 0-indexed 0..M-1)

N            = number of physical memory cells in the row (0-indexed 0..N-1)
M            = number of code options in the catalog
LMAX         = the checker will exhaustively test every contiguous physical burst
               (a "multi-bit upset" that hits LEN physically-adjacent cells at once,
               for every LEN in 1..LMAX and every starting offset) against whatever
               codeword-partition + code choice the solution proposes.
w,t,cost     = a code option: word length w (cells per codeword), correction
               capability t (max simultaneous flipped cells inside ONE codeword that
               the code can still correct), decoder cost `cost` (an abstract op-count
               / decoder-latency surrogate -- lower is cheaper to decode).
"""
import sys


# (N, LMAX, word-length menu) per testId -- a difficulty/trap ladder.
# testId 1-2: LMAX small (near 1) -> logical/contiguous layout is not really a trap.
# testId 3-10: LMAX grows relative to N -> a genuine multi-cell physical burst that a
#              contiguous ("logical order") mapping cannot spread out, but a physically
#              interleaved mapping can, at a much weaker (cheaper) correction strength.
PLAN = {
    1: (16, 1, [2, 4, 8]),
    2: (24, 2, [2, 3, 4]),
    3: (32, 4, [2, 4]),
    4: (32, 6, [2, 4]),
    5: (48, 6, [2, 3, 4, 6]),
    6: (48, 8, [2, 3, 4, 6, 8]),
    7: (64, 8, [2, 4, 8]),
    8: (64, 10, [2, 4, 8]),
    9: (96, 12, [2, 3, 4, 6, 8]),
    10: (128, 16, [2, 4, 8]),
}


FIXED = 50  # per-codeword fixed decoder overhead (routing / check-bit generation)


def build_catalog(testId, N, LMAX, wlist):
    entries = []  # (w, t, cost)
    seen = set()
    for w in wlist:
        assert N % w == 0
        topts = sorted(set(list(range(1, 17)) + [20, 24, 32, 48, 64, w]))
        topts = [t for t in topts if 1 <= t <= w]
        for t in topts:
            key = (w, t)
            if key in seen:
                continue
            seen.add(key)
            jitter = (testId * 31 + w * 7 + t * 3) % 5
            # `+FIXED` models a per-codeword routing/check-bit overhead that is paid no
            # matter how weak the code is -- it discourages splitting into an unbounded
            # number of tiny codewords purely to chase cheap interleaving.
            cost = FIXED + w + 8 * t * t + jitter
            entries.append((w, t, cost))
    # A dedicated whole-row option: one giant codeword spanning all N cells, strong
    # enough to correct any burst up to LMAX (t=LMAX suffices since no window ever
    # exceeds LMAX cells). Always present so a `strong`/`greedy` search always has a
    # last-resort fallback and the instance can never be unsolvable.
    safe_key = (N, LMAX)
    if safe_key not in seen:
        entries.append((N, LMAX, FIXED + N + 8 * LMAX * LMAX))
    return entries


def main():
    testId = int(sys.argv[1])
    testId = ((testId - 1) % 10) + 1
    N, LMAX, wlist = PLAN[testId]
    catalog = build_catalog(testId, N, LMAX, wlist)
    M = len(catalog)
    out = [f"{N} {M} {LMAX}"]
    for w, t, cost in catalog:
        out.append(f"{w} {t} {cost}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
