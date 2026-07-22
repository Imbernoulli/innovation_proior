#!/usr/bin/env python3
"""Deterministic checker for Privacy-Preserving Sensor Activation Codes.

Usage: python3 verify.py <in> <out> <ans>

The checker validates a constant-weight binary code with pairwise overlap and 2-cover-free
constraints. On any feasibility violation it prints Ratio: 0.0 and exits 0. Otherwise it
scores the number of rows against the same internal first-fit reference used by the
trivial solution.
"""
import sys

MASK64 = (1 << 64) - 1


def fail(reason):
    print("INVALID (%s) Ratio: 0.0" % reason)
    sys.exit(0)


class XorShift64:
    def __init__(self, seed):
        self.s = (seed ^ 0x9E3779B97F4A7C15) & MASK64
        if self.s == 0:
            self.s = 0x123456789ABCDEF

    def next(self):
        x = self.s
        x ^= (x << 13) & MASK64
        x ^= (x >> 7)
        x ^= (x << 17) & MASK64
        self.s = x & MASK64
        return self.s


def parse_uint(tok, name):
    if not tok or len(tok) > 20:
        fail("bad integer %s" % name)
    for ch in tok:
        if ch < "0" or ch > "9":
            fail("bad integer %s" % name)
    return int(tok)


def parse_instance(path):
    toks = open(path, "r", encoding="ascii").read().split()
    if len(toks) != 6:
        fail("bad instance token count")
    vals = [parse_uint(t, "instance") for t in toks]
    n, w, lam, d, cap, salt = vals
    if not (8 <= n <= 80 and 1 <= w <= n and 0 <= lam <= w and d == 2 and 0 <= cap <= 500):
        fail("bad instance bounds")
    return n, w, lam, d, cap, salt


def sample_mask(rng, n, w):
    arr = list(range(n))
    for i in range(w):
        j = i + (rng.next() % (n - i))
        arr[i], arr[j] = arr[j], arr[i]
    mask = 0
    for i in range(w):
        mask |= 1 << arr[i]
    return mask


def can_add(code, cand, lam):
    m = len(code)
    for row in code:
        if (cand & row).bit_count() > lam:
            return False

    # The new row must not be covered by the union of two old rows.
    for i in range(m):
        rem = cand & ~code[i]
        for j in range(i + 1, m):
            if (rem & ~code[j]) == 0:
                return False

    # No old row may become covered by the new row and one other old row.
    for i in range(m):
        rem = code[i] & ~cand
        for j in range(m):
            if i != j and (rem & ~code[j]) == 0:
                return False
    return True


def reference_code(n, w, lam, cap, salt):
    rng = XorShift64(0xB00 + 131 * salt)
    attempts = 100 + 3 * n
    seen = set()
    candidates = []
    for _ in range(attempts):
        cand = sample_mask(rng, n, w)
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)

    code = []
    for cand in candidates:
        if len(code) >= cap:
            break
        if can_add(code, cand, lam):
            code.append(cand)
    return code


def row_to_mask(tok, n, w):
    if len(tok) != n:
        fail("row has wrong length")
    mask = 0
    wt = 0
    for j, ch in enumerate(tok):
        if ch == "1":
            mask |= 1 << j
            wt += 1
        elif ch != "0":
            fail("row is not binary")
    if wt != w:
        fail("row has wrong weight")
    return mask


def validate_rows(rows, lam):
    m = len(rows)
    if len(set(rows)) != m:
        fail("duplicate rows")

    for i in range(m):
        for j in range(i + 1, m):
            if (rows[i] & rows[j]).bit_count() > lam:
                fail("pairwise overlap cap violated")

    for i in range(m):
        target = rows[i]
        others = [rows[j] for j in range(m) if j != i]
        for a in range(len(others)):
            rem = target & ~others[a]
            for b in range(a + 1, len(others)):
                if (rem & ~others[b]) == 0:
                    fail("2-cover-free constraint violated")


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    n, w, lam, _d, cap, salt = parse_instance(sys.argv[1])

    max_bytes = max(10000, (cap + 1) * (n + 4) + 1000)
    with open(sys.argv[2], "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        fail("output too large")
    try:
        raw = data.decode("ascii")
    except UnicodeDecodeError:
        fail("output is not ascii")
    low = raw.lower()
    if "nan" in low or "inf" in low:
        fail("non-finite token")

    toks = raw.split()
    if not toks:
        fail("empty output")
    m = parse_uint(toks[0], "m")
    if m > cap:
        fail("m exceeds row cap")
    if len(toks) != 1 + m:
        fail("wrong token count")

    rows = [row_to_mask(tok, n, w) for tok in toks[1:]]
    validate_rows(rows, lam)

    F = len(rows)
    B = len(reference_code(n, w, lam, cap, salt))
    if B <= 0:
        B = 1
    sc = min(1000.0, 100.0 * float(F) / max(1e-9, float(B)))
    print("valid. F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
