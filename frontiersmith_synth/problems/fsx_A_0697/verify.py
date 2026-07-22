#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for lifetime-shelf-packer.

Feasibility: N integer offsets, non-negative & bounded, no address-range
overlap between crates whose stays overlap in time.
Objective (minimize): F = peak + LAMBDA * distinct_aisles_touched.
Baseline B: the checker's own never-reuse "bump" placement.
"""
import sys

ADDR_CAP = 10 ** 12


def fail(msg):
    print("INVALID: %s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        in_tokens = f.read().split()
    pos = 0

    def next_int():
        nonlocal pos
        v = int(in_tokens[pos])
        pos += 1
        return v

    N = next_int()
    M = next_int()
    PAGE = next_int()
    LAMBDA = next_int()

    sizes = [0] * N
    births = [0] * N
    deaths = [0] * N
    for i in range(N):
        sizes[i] = next_int()
        births[i] = next_int()
        deaths[i] = next_int()

    checks = []
    for _ in range(M):
        t = next_int()
        c = next_int()
        checks.append(c - 1)

    with open(out_path) as f:
        out_tokens = f.read().split()

    if len(out_tokens) != N:
        fail("expected %d tokens, got %d" % (N, len(out_tokens)))

    addr = [0] * N
    for i in range(N):
        tok = out_tokens[i]
        try:
            v = int(tok)
        except ValueError:
            fail("token %d (%r) is not an integer" % (i, tok))
        if v < 0 or v > ADDR_CAP:
            fail("address %d out of range" % v)
        addr[i] = v

    # Feasibility: pairwise overlap check among crates with overlapping stays.
    for i in range(N):
        bi, di, si, ai = births[i], deaths[i], sizes[i], addr[i]
        for j in range(i + 1, N):
            bj, dj, sj, aj = births[j], deaths[j], sizes[j], addr[j]
            if bi < dj and bj < di:  # stays overlap in time
                if ai < aj + sj and aj < ai + si:  # address ranges overlap
                    fail("crates %d and %d overlap in both time and space" % (i, j))

    def score(addr_assign):
        peak = 0
        for i in range(N):
            end = addr_assign[i] + sizes[i]
            if end > peak:
                peak = end
        pages = set()
        for c in checks:
            a = addr_assign[c]
            s = sizes[c]
            p0 = a // PAGE
            p1 = (a + s - 1) // PAGE
            for p in range(p0, p1 + 1):
                pages.add(p)
        return peak + LAMBDA * len(pages)

    F = score(addr)

    bump_addr = [0] * N
    running = 0
    for i in range(N):
        bump_addr[i] = running
        running += sizes[i]
    B = score(bump_addr)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
