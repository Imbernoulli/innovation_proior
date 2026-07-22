#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is unused, per format-C contract)

Scores a keyslot assignment (permutation of symbols -> keyboard slots) on the
"one keyboard for three quarrelling languages" objective:

  For each language k, the finger-travel cost TC_k is the digraph-frequency-
  weighted sum of Euclidean travel between the slots assigned to consecutive
  symbols. TC_k is normalized by that language's own random-layout baseline
  (closed-form expected cost under a uniform-random permutation) to get a
  per-language normalized cost NC_k. The graded objective is the ENVELOPE
  F = max_k NC_k -- the worst-served language, not the pooled average.

This is a MINIMIZATION objective: lower F is better. The checker's own
"trivial feasible construction" baseline B is F under the identity assignment
(slot[i] = i), and the printed Ratio is min(1000, 100*B/F) / 1000.
"""
import sys
import math


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        n = int(next(it)); k = int(next(it))
        coords = []
        for _ in range(n):
            x = float(next(it)); y = float(next(it))
            coords.append((x, y))
        freq = []
        for _lang in range(k):
            mat = []
            for _i in range(n):
                row = [int(next(it)) for _ in range(n)]
                mat.append(row)
            freq.append(mat)
    except StopIteration:
        raise ValueError("truncated input")
    return n, k, coords, freq


def travel_matrix(n, coords):
    T = [[0.0] * n for _ in range(n)]
    for u in range(n):
        xu, yu = coords[u]
        for v in range(n):
            if u == v:
                continue
            xv, yv = coords[v]
            T[u][v] = math.hypot(xu - xv, yu - yv)
    return T


def language_stats(n, k, freq, T):
    """Return (S_k list, T_avg, baseline_k list)."""
    tot = 0.0
    cnt = 0
    for u in range(n):
        for v in range(n):
            if u != v:
                tot += T[u][v]
                cnt += 1
    t_avg = tot / cnt if cnt else 0.0

    s = []
    for lang in range(k):
        total = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total += freq[lang][i][j]
        s.append(total)
    baseline = [t_avg * sk for sk in s]
    return s, t_avg, baseline


def envelope_cost(perm, n, k, freq, T, baseline):
    ncs = []
    for lang in range(k):
        tc = 0.0
        fk = freq[lang]
        for i in range(n):
            pi = perm[i]
            row = fk[i]
            for j in range(n):
                if i == j:
                    continue
                w = row[j]
                if w:
                    tc += w * T[pi][perm[j]]
        b = baseline[lang]
        ncs.append(tc / b if b > 1e-9 else 0.0)
    return max(ncs), ncs


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        n, k, coords, freq = read_input(in_path)
    except Exception as e:
        print(f"BAD_INPUT: {e}")
        print("Ratio: 0.0")
        sys.exit(0)

    try:
        with open(out_path, "r") as f:
            raw_toks = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw_toks) != n:
        fail(f"expected exactly {n} tokens, got {len(raw_toks)}")

    perm = []
    for tok in raw_toks:
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token: {tok!r}")
        except OverflowError:
            fail(f"token out of representable range: {tok!r}")
        # v is a Python int here (arbitrary precision, always finite by
        # construction -- nan/inf/1.5 already failed the int() parse above).
        # Do the range check with plain int comparison (never coerce to
        # float / call math.isfinite on it: that OVERFLOWS on a huge-digit
        # token instead of failing cleanly).
        if v < 0 or v >= n:
            fail(f"slot index out of range [0, n-1]: {tok!r}")
        perm.append(v)

    if len(set(perm)) != n:
        fail("not a bijection (duplicate slot assignment)")

    T = travel_matrix(n, coords)
    _, _, baseline = language_stats(n, k, freq, T)

    F, per_lang = envelope_cost(perm, n, k, freq, T, baseline)

    identity = list(range(n))
    B, _ = envelope_cost(identity, n, k, freq, T, baseline)

    if B <= 1e-9:
        fail("degenerate instance (zero baseline)")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("envelope F=%.6f baseline B=%.6f per_lang=%s" % (F, B, ["%.4f" % x for x in per_lang]))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
