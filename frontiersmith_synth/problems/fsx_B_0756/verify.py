#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the sleeper-train
overbooking-compensation-ladder problem. Prints 'Ratio: <float in [0,1]>' and exits 0.
"""
import math
import sys

INF = float("inf")
BASELINE_FRAC = 0.15


def fail(msg):
    print("INFEASIBLE:", msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(path):
    try:
        with open(path, "r") as f:
            txt = f.read()
    except Exception as e:
        fail(f"cannot read {path}: {e}")
    toks = txt.split()
    out = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            fail(f"non-integer token {t!r}")
        if abs(v) > 10 ** 12:
            fail(f"token out of range {t!r}")
        out.append(v)
    return out


def simulate(sold, ladder, nights):
    """nights: list of (capacity, fare, max_sell, noshow[], threshold[])
    sold: list of chosen SOLD_i (already range-validated by caller)
    ladder: sorted increasing 5 ints
    Returns total net score (revenue - compensation - penalties)."""
    total = 0.0
    top = ladder[-1]
    for (capacity, fare, max_sell, noshow, threshold), s in zip(nights, sold):
        revenue = s * fare
        shown_idx = [j for j in range(s) if noshow[j] == 0]
        shows = len(shown_idx)
        comp = 0.0
        invol = 0
        if shows > capacity:
            overflow = shows - capacity

            def cost_of(j):
                th = threshold[j]
                if th > top:
                    return INF
                for step in ladder:
                    if step >= th:
                        return step
                return INF  # unreachable, defensive

            ranked = sorted(shown_idx, key=lambda j: (cost_of(j), j))
            bumped = ranked[:overflow]
            for j in bumped:
                c = cost_of(j)
                if c == INF:
                    invol += 1
                else:
                    comp += c
        total += revenue - comp - invol * PENALTY_GLOBAL[0]
    return total


PENALTY_GLOBAL = [900]  # patched from input in main()


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = read_ints(in_path)
    if not itoks:
        fail("empty input")
    pos = 0
    n = itoks[pos]; pos += 1
    if n <= 0:
        fail("bad N in input")
    if pos + 3 > len(itoks):
        fail("truncated header")
    ladder_lo, ladder_hi, penalty = itoks[pos], itoks[pos + 1], itoks[pos + 2]
    pos += 3
    PENALTY_GLOBAL[0] = penalty

    nights = []
    for _ in range(n):
        if pos + 3 > len(itoks):
            fail("truncated night header")
        capacity, fare, max_sell = itoks[pos], itoks[pos + 1], itoks[pos + 2]
        pos += 3
        if capacity <= 0 or fare <= 0 or max_sell < capacity:
            fail("bad night header")
        if pos + 2 * max_sell > len(itoks):
            fail("truncated passenger data")
        noshow = []
        threshold = []
        for _ in range(max_sell):
            ns = itoks[pos]; th = itoks[pos + 1]; pos += 2
            if ns not in (0, 1) or th <= 0:
                fail("bad passenger record")
            noshow.append(ns)
            threshold.append(th)
        nights.append((capacity, fare, max_sell, noshow, threshold))

    # ---- parse participant output ----
    otoks = read_ints(out_path)
    if len(otoks) < 5 + n:
        fail(f"expected >= {5+n} output tokens, got {len(otoks)}")
    ladder = otoks[:5]
    sold = otoks[5:5 + n]

    for k in range(5):
        if not (ladder_lo <= ladder[k] <= ladder_hi):
            fail(f"ladder step {k} = {ladder[k]} out of [{ladder_lo},{ladder_hi}]")
    for k in range(4):
        if not (ladder[k] < ladder[k + 1]):
            fail("ladder not strictly increasing")

    for i in range(n):
        capacity, fare, max_sell, _, _ = nights[i]
        if not (0 <= sold[i] <= max_sell):
            fail(f"night {i}: sold={sold[i]} out of [0,{max_sell}]")

    F = simulate(sold, ladder, nights)

    # ---- checker's own weak baseline B: sell only BASELINE_FRAC of capacity, never oversell ----
    base_sold = [max(1, round(c * BASELINE_FRAC)) for (c, f, m, ns, th) in nights]
    base_ladder = [ladder_lo, ladder_lo + 1, ladder_lo + 2, ladder_lo + 3, ladder_lo + 4]
    B = simulate(base_sold, base_ladder, nights)
    if B <= 0:
        fail("internal baseline non-positive (should not happen)")

    sc = min(1000.0, 100.0 * max(0.0, F) / max(1e-9, B))
    ratio = sc / 1000.0
    print(f"F={F:.3f} B={B:.3f}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
