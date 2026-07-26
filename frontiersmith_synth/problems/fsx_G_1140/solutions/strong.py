# TIER: strong
"""
Insight: exploit the planted 2-lane min-plus structure directly instead of
fitting one blended additive rate.

1. Pure-'b' training strings can NEVER switch lanes -> their cost is exactly
   p_b*length. Average cost/length over them gives p_b cleanly.
2. Pure-'a' training strings trace out a genuine changepoint curve: for short
   lengths the "never switch" lane dominates (slope p_a through the origin),
   for long lengths the "switch immediately" lane dominates (a line with
   intercept q-r_a and slope r_a). Try every split point, fit both segments,
   and keep the split with least total squared residual -- a direct,
   generalizable regime-change detector. This recovers p_a, r_a and q.
3. r_b never shows up in isolation (mixing with 'b' never switches lanes by
   itself), so it is recovered by a small 1-D search over general mixed
   training strings: pick the r_b that makes the FULL two-state automaton
   (with the already-recovered p_a,p_b,q,r_a) best reproduce their costs.

The result is emitted as a genuine 2-state min-plus automaton, which is the
mechanism itself rather than a fit to it, so it generalizes to far longer
held-out strings.
"""
import sys


def true_cost(w, p_a, p_b, q, r_a, r_b):
    INF = float("inf")
    d0, d1 = 0.0, INF
    for ch in w:
        if ch == "a":
            n0 = d0 + p_a
            n1 = min(d0 + q, d1 + r_a)
        else:
            n0 = d0 + p_b
            n1 = d1 + r_b
        d0, d1 = n0, n1
    return min(d0, d1)


def main():
    data = sys.stdin.read().split("\n")
    first = data[0].split()
    n = int(first[1])
    rows = []
    for i in range(2, 2 + n):
        parts = data[i].split()
        s, c = parts[0], int(parts[1])
        rows.append((s, c))

    pure_a = sorted([(len(s), c) for s, c in rows if s and set(s) == {"a"}])
    pure_b = sorted([(len(s), c) for s, c in rows if s and set(s) == {"b"}])
    mixed = [(s, c) for s, c in rows if "a" in s and "b" in s]

    if pure_b:
        p_b_est = sum(c / L for L, c in pure_b) / len(pure_b)
    else:
        p_b_est = sum(c for s, c in rows) / max(1.0, sum(len(s) for s, c in rows))

    best = None
    Ls = [L for L, c in pure_a]
    m = len(pure_a)
    if m >= 3:
        for split in range(1, m):
            short = pure_a[:split]
            long_ = pure_a[split:]
            if len(long_) < 2:
                continue
            p_a_c = sum(c / L for L, c in short) / len(short)
            n2 = len(long_)
            sx = sum(L for L, c in long_)
            sy = sum(c for L, c in long_)
            sxx = sum(L * L for L, c in long_)
            sxy = sum(L * c for L, c in long_)
            denom = n2 * sxx - sx * sx
            if abs(denom) < 1e-9:
                continue
            mslope = (n2 * sxy - sx * sy) / denom
            bint = (sy - mslope * sx) / n2
            res = 0.0
            for L, c in pure_a:
                pred = p_a_c * L if L < Ls[split] else mslope * L + bint
                res += (pred - c) ** 2
            if best is None or res < best[0]:
                best = (res, p_a_c, mslope, bint)

    if best is not None:
        _, p_a_est, r_a_est, b_est = best
        q_est = b_est + r_a_est
        r_a_est = max(0.0, r_a_est)
        q_est = max(0.0, q_est)
    else:
        # fallback: not enough pure-a data, assume no benefit from switching
        p_a_est = sum(c / L for L, c in pure_a) / max(1, len(pure_a)) if pure_a else p_b_est
        r_a_est = p_a_est
        q_est = 1e5

    if mixed:
        best_rb = None
        lo = 1
        hi = max(1, int(round(p_b_est)) + 5)
        for r_b_cand in range(lo, hi + 1):
            err = 0.0
            for s, c in mixed:
                pred = true_cost(s, p_a_est, p_b_est, q_est, r_a_est, r_b_cand)
                err += (pred - c) ** 2
            if best_rb is None or err < best_rb[0]:
                best_rb = (err, r_b_cand)
        r_b_est = float(best_rb[1])
    else:
        r_b_est = p_b_est

    out = []
    out.append("2 5")
    out.append("0 a 0 %.6f" % p_a_est)
    out.append("0 a 1 %.6f" % q_est)
    out.append("0 b 0 %.6f" % p_b_est)
    out.append("1 a 1 %.6f" % r_a_est)
    out.append("1 b 1 %.6f" % r_b_est)
    out.append("0")
    out.append("2")
    out.append("0 0.0")
    out.append("1 0.0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
