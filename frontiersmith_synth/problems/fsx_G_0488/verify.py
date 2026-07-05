#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (ans ignored)

Deterministic scorer for the "large progression-free set in Z_p" problem.

Reads the prime p from <in> and the participant's set S from <out>. Validates
feasibility STRICTLY (integer tokens only, in-range, distinct, progression-free
mod p) and, on any violation, prints `Ratio: 0.0` and exits 0. Otherwise scores

    Ratio = min(1.0, 0.1 * |S| / |B|)

where B is a valid baseline set the checker constructs itself (the base-3
"no-carry" set on a short register), guaranteeing a positive normalizer.
"""
import sys

# Hard caps so an adversarial / flooded output cannot blow up the checker.
MAX_OUT_BYTES = 4_000_000     # refuse to read more than a few MB
MAX_ELEMS = 12_000            # any legitimate set here is < ~1500 elements


def die0(reason: str) -> None:
    print("reason:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def base3_nocarry(bound: int):
    """All non-negative integers < bound whose base-3 representation uses only
    the digits 0 and 1 (a classic progression-free set over the integers)."""
    if bound <= 0:
        return []
    pows = []
    v = 1
    while v < bound:
        pows.append(v)
        v *= 3
    n = len(pows)
    out = []
    for mask in range(1 << n):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += pows[idx]
            m >>= 1
            idx += 1
        if s < bound:
            out.append(s)
    return sorted(set(out))


def read_int(path: str) -> int:
    with open(path, "r") as f:
        txt = f.read(1_000_000)
    toks = txt.split()
    if not toks:
        die0("empty input")
    return int(toks[0])


def parse_set(path: str, p: int):
    with open(path, "rb") as f:
        raw = f.read(MAX_OUT_BYTES + 1)
    if len(raw) > MAX_OUT_BYTES:
        die0("output too large")
    try:
        txt = raw.decode("utf-8")
    except Exception:
        die0("non-utf8 output")
    toks = txt.split()
    if len(toks) > MAX_ELEMS:
        die0("too many tokens")
    S = []
    seen = set()
    for t in toks:
        # strict integer parse: rejects floats, 'nan', 'inf', hex, etc.
        try:
            x = int(t)
        except ValueError:
            die0("non-integer token: %r" % t)
        if x < 0 or x >= p:
            die0("element out of range [0,p): %d" % x)
        if x in seen:
            die0("duplicate element: %d" % x)
        seen.add(x)
        S.append(x)
    return S


def is_progression_free(S, p: int) -> bool:
    """True iff no a != c in S has its modular midpoint (a+c)*inv2 also in S."""
    Sset = set(S)
    inv2 = pow(2, p - 2, p)  # 2^{-1} mod p (p odd prime)
    Sl = sorted(S)
    n = len(Sl)
    for i in range(n):
        a = Sl[i]
        for j in range(i + 1, n):
            mid = ((a + Sl[j]) * inv2) % p
            if mid in Sset:
                return False
    return True


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>", file=sys.stderr)
        sys.exit(2)
    p = read_int(sys.argv[1])
    if p < 3 or p % 2 == 0:
        die0("bad prime p")

    S = parse_set(sys.argv[2], p)

    if not is_progression_free(S, p):
        die0("set contains a 3-term arithmetic progression mod p")

    # Internal baseline: no-carry base-3 set on a short register (< (p/2)/9),
    # always progression-free mod p and strictly positive.
    half = (p + 1) // 2
    B = base3_nocarry(max(2, half // 9))
    b = max(1, len(B))

    F = len(S)
    sc = min(1000.0, 100.0 * F / max(1e-9, float(b)))
    ratio = sc / 1000.0
    print("p=%d |S|=%d baseline=%d" % (p, F, b))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
