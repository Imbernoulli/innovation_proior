#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for Gray-Adjacent State Encoding.

Validates the participant's state -> B-bit codeword bijection, rebuilds the B
next-state-bit Boolean functions (each over B+K variables: B current-codeword bits +
K=2 symbol bits), minimizes each to a canonical sum-of-products via an essential-prime-
implicant + deterministic-greedy cover (a standard, fast, fully deterministic heuristic;
always produces a semantically-correct SOP, i.e. equivalence is exact by construction),
counts total literals F, and compares against the checker's own baseline B0 = literal
count under the naive identity encoding (codeword(u) = binary(u)).

Ratio = min(1, 0.1 * B0 / F)   (minimization; fewer literals is better)
"""
import sys
import math


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        N = int(next(it))
        K = int(next(it))
    except StopIteration:
        raise ValueError("truncated input header")
    M = 1 << K
    trans = []
    for _u in range(N):
        row = []
        for _s in range(M):
            row.append(int(next(it)))
        trans.append(row)
    return N, K, M, trans


def parse_output(path, N, B):
    """Return list codeword[u] (int) or None on any violation."""
    try:
        with open(path) as f:
            lines = f.read().split("\n")
    except Exception:
        return None
    # strip trailing blank lines
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != N:
        return None
    seen = set()
    codeword = [0] * N
    for u, line in enumerate(lines):
        tok = line.strip()
        if len(tok) != B:
            return None
        if any(c not in "01" for c in tok):
            return None
        val = int(tok, 2)
        if val in seen:
            return None
        seen.add(val)
        codeword[u] = val
    if len(seen) != N:
        return None
    return codeword


def bits_of(x, n):
    return tuple((x >> i) & 1 for i in range(n))


def build_truth_tables(N, K, M, trans, codeword, B):
    """codeword: list[u] -> int in [0, 2**B).  N == 2**B (bijective onto full cube).
    Returns list of B truth tables, each a dict from tuple(bits, length B+K) -> 0/1."""
    decode = [None] * N  # codeword value -> state id (bijective since N == 2**B)
    for u, c in enumerate(codeword):
        decode[c] = u

    totalvars = B + K
    tables = [dict() for _ in range(B)]
    for cw in range(N):
        u = decode[cw]
        state_bits = bits_of(cw, B)
        for s in range(M):
            sym_bits = bits_of(s, K)
            key = state_bits + sym_bits
            nxt = trans[u][s]
            nxt_bits = bits_of(codeword[nxt], B)
            for i in range(B):
                tables[i][key] = nxt_bits[i]
    return tables, totalvars


def covered_mask(fixed_pos, fixed_val, dash_pos, n):
    """Bitmask (over 2**n rows, row index = int(bits) with bit0=LSB=var0) of all rows
    consistent with fixing fixed_pos[i]=fixed_val[i] and leaving dash_pos free."""
    base = 0
    for p, v in zip(fixed_pos, fixed_val):
        if v:
            base |= (1 << p)
    mask = 0
    nd = len(dash_pos)
    for combo in range(1 << nd):
        idx = base
        for k, p in enumerate(dash_pos):
            if (combo >> k) & 1:
                idx |= (1 << p)
        mask |= (1 << idx)
    return mask


def literal_count_for_table(table, n):
    """table: dict tuple(bits,len n) -> 0/1, fully specified (2**n entries).
    Returns total literal count of an essential-PI + deterministic-greedy SOP cover
    of the ON-set (function == 1). If ON-set is empty, 0 literals (constant-0 output,
    zero product terms). If ON-set is everything, the single all-dash implicant covers
    it with 0 literals (constant-1 output)."""
    on_mask = 0
    for key, val in table.items():
        if val:
            row = 0
            for k, b in enumerate(key):
                if b:
                    row |= (1 << k)
            on_mask |= (1 << row)
    if on_mask == 0:
        return 0

    all_positions = list(range(n))
    # enumerate ALL patterns (fixed subset + values) that are valid implicants
    # (covered rows subset of on_mask); n <= 7 keeps this fast (<= 4**n row-visits).
    implicants = []  # (fixed_pos tuple, fixed_val tuple, mask)
    from itertools import combinations, product as iproduct

    for d in range(0, n + 1):
        # d = number of dash (free) positions; fixed = n-d positions get 0/1
        for dash_pos in combinations(all_positions, d):
            dash_set = set(dash_pos)
            fixed_pos = tuple(p for p in all_positions if p not in dash_set)
            for fixed_val in iproduct((0, 1), repeat=len(fixed_pos)):
                m = covered_mask(fixed_pos, fixed_val, dash_pos, n)
                if m & on_mask == m:  # subset of ON-set
                    implicants.append((fixed_pos, fixed_val, dash_pos, m))

    # primality filter: an implicant is prime iff no valid implicant with one MORE
    # dash (a direct generalization dropping exactly one fixed position) exists.
    by_mask = {}
    for fp, fv, dp, m in implicants:
        by_mask.setdefault((tuple(sorted(dp)),), []).append((fp, fv, m))

    valid_set = set()
    for fp, fv, dp, m in implicants:
        valid_set.add((fp, fv))

    prime = []
    for fp, fv, dp, m in implicants:
        is_prime = True
        for drop_idx in range(len(fp)):
            new_fixed_pos = fp[:drop_idx] + fp[drop_idx + 1:]
            new_fixed_val = fv[:drop_idx] + fv[drop_idx + 1:]
            if (new_fixed_pos, new_fixed_val) in valid_set:
                is_prime = False
                break
        if is_prime:
            prime.append((fp, fv, m, len(fp)))

    # canonical deterministic order
    prime.sort(key=lambda t: (t[3], t[0], t[1]))

    # minterms to cover = individual set bits of on_mask
    minterms = [b for b in range(1 << n) if (on_mask >> b) & 1]

    cover_of = []  # for each minterm, list of prime-implicant indices covering it
    idx_by_minterm = {mt: [] for mt in minterms}
    for pi_idx, (fp, fv, m, nlit) in enumerate(prime):
        b = m
        pos = 0
        while b:
            if b & 1:
                idx_by_minterm[pos].append(pi_idx)
            b >>= 1
            pos += 1

    selected = set()
    covered = 0
    # essential prime implicants: minterms covered by exactly one PI
    for mt in minterms:
        covers = idx_by_minterm[mt]
        if len(covers) == 1:
            selected.add(covers[0])
    for pi_idx in selected:
        covered |= prime[pi_idx][2]

    # deterministic greedy cover for the remainder
    remaining = on_mask & ~covered
    while remaining:
        best_idx = None
        best_gain = -1
        for pi_idx, (fp, fv, m, nlit) in enumerate(prime):
            if pi_idx in selected:
                continue
            gain = bin(m & remaining).count("1")
            if gain > best_gain:
                best_gain = gain
                best_idx = pi_idx
            elif gain == best_gain and best_idx is not None:
                if (fp, fv) < (prime[best_idx][0], prime[best_idx][1]):
                    best_idx = pi_idx
        if best_idx is None or best_gain <= 0:
            break
        selected.add(best_idx)
        remaining &= ~prime[best_idx][2]

    total_literals = sum(prime[pi_idx][3] for pi_idx in selected)
    return total_literals


def total_literal_cost(N, K, M, trans, codeword, B):
    tables, totalvars = build_truth_tables(N, K, M, trans, codeword, B)
    return sum(literal_count_for_table(t, totalvars) for t in tables)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]

    N, K, M, trans = read_input(in_path)
    B = N.bit_length() - 1
    if (1 << B) != N:
        fail("N is not a power of two (generator contract violated)")

    codeword = parse_output(out_path, N, B)
    if codeword is None:
        fail("output must be exactly N lines, each a distinct length-B binary string")

    F = total_literal_cost(N, K, M, trans, codeword, B)

    identity_codeword = list(range(N))
    B0 = total_literal_cost(N, K, M, trans, identity_codeword, B)
    B0 = max(1, B0)

    score = min(1000.0, 100.0 * B0 / max(1e-9, F))
    ratio = score / 1000.0
    print(f"F={F} B0={B0}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
