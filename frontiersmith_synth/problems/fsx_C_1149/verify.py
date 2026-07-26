#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is an unused placeholder, per format-C contract)

Deterministic scorer for "Quarry Foreman" (epoch-boundary-carver).

Instance:
  L Q BASE
  h_0 h_1 ... h_{L-1}          (decoy hardness profile, not used for scoring)
  l_1 r_1 w_1
  ...
  l_Q r_Q w_Q

Artifact (stdout):
  m
  b_1 b_2 ... b_m              (interior cut positions, strictly increasing, 0 < b_j < L)

The m cuts, plus the implicit ends 0 and L, define k = m+1 slabs
[0,b_1), [b_1,b_2), ..., [b_m,L).

  BUILD = sum over slabs of ( BASE + size^GAMMA )         GAMMA = 1.5 fixed
  TOUCH = sum over queries i of  w_i * (number of slabs overlapping [l_i, r_i))
        = sum_i w_i  +  sum_{interior cut b} pen(b)
    where pen(b) = sum_{i : l_i < b < r_i} w_i   (a cut costs pen(b) once,
    however many queries straddle it -- purely a function of where it sits,
    independent of any other cut)

  F = BUILD + TOUCH        (minimize)

Baseline B = F under the single-slab (m=0, no cuts) construction.
Ratio = min(1000, 100*B/F) / 1000    (F=0 impossible since BASE>0)

Pure float arithmetic with a fixed exponent, O((L) + Q log Q); deterministic.
"""
import sys
from bisect import bisect_left, bisect_right

GAMMA = 1.5


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        L = int(next(it))
        Q = int(next(it))
        BASE = int(next(it))
        h = [int(next(it)) for _ in range(L)]
        queries = []
        for _ in range(Q):
            l = int(next(it))
            r = int(next(it))
            w = int(next(it))
            queries.append((l, r, w))
    except StopIteration:
        raise ValueError("truncated input")
    if L <= 1 or Q <= 0 or BASE <= 0:
        raise ValueError("bad header")
    for (l, r, w) in queries:
        if not (0 <= l < r <= L) or w <= 0:
            raise ValueError("bad query")
    return L, Q, BASE, h, queries


def total_cost(L, BASE, queries, sum_w, boundaries):
    """boundaries: sorted list of strictly-interior cut points (may be empty)."""
    # ---- BUILD ----
    build = 0.0
    prev = 0
    for b in boundaries:
        size = b - prev
        build += BASE + size ** GAMMA
        prev = b
    size = L - prev
    build += BASE + size ** GAMMA

    # ---- TOUCH ----
    touch_extra = 0.0
    for (l, r, w) in queries:
        lo = bisect_right(boundaries, l)   # first index with b > l
        hi = bisect_left(boundaries, r)    # first index with b >= r
        cnt = hi - lo
        if cnt > 0:
            touch_extra += w * cnt
    touch = sum_w + touch_extra

    return build + touch


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        L, Q, BASE, h, queries = read_instance(in_path)
    except Exception as e:
        print(f"BAD_INPUT: {e}")
        print("Ratio: 0.0")
        sys.exit(0)

    sum_w = sum(w for (_, _, w) in queries)

    try:
        with open(out_path, "r") as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if not raw:
        fail("empty output")

    try:
        m = int(raw[0])
    except ValueError:
        fail(f"non-integer m token: {raw[0]!r}")
    except OverflowError:
        fail(f"m token out of range: {raw[0]!r}")

    if m < 0 or m > L - 1:
        fail(f"m={m} out of range [0, {L - 1}]")
    if len(raw) != 1 + m:
        fail(f"expected {1 + m} tokens, got {len(raw)}")

    boundaries = []
    for tok in raw[1:]:
        try:
            v = int(tok)  # rejects 'nan', 'inf', '3.5', '1e3', ...
        except ValueError:
            fail(f"non-integer cut token: {tok!r}")
        except OverflowError:
            fail(f"cut token out of representable range: {tok!r}")
        if v < 1 or v > L - 1:
            fail(f"cut {v} out of range (0, {L})")
        boundaries.append(v)

    for j in range(1, len(boundaries)):
        if boundaries[j] <= boundaries[j - 1]:
            fail("cuts must be strictly increasing")

    F = total_cost(L, BASE, queries, sum_w, boundaries)
    B = total_cost(L, BASE, queries, sum_w, [])
    if B <= 0:
        fail("degenerate instance (zero baseline)")
    if not (F == F) or F in (float("inf"), float("-inf")) or F <= 0:
        fail("non-finite or non-positive objective")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"F={F:.4f} B={B:.4f} m={m}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
