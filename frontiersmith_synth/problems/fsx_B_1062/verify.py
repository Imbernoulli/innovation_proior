#!/usr/bin/env python3
# Deterministic checker for Relay Throughput Spacing (format C, maximize the
# minimum end-to-end throughput across source/destination pairs).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1] on its own final line, exits 0.
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_path(S, D, relays):
    return [S] + relays + [D]


def bottleneck_rates(paths, P, alpha, N0):
    """Global TDMA-round schedule: in round r, the node at index r-1 of every
    pair whose path has >= r hops transmits to the node at index r of that
    same pair. All simultaneously-transmitting pairs interfere with each
    other's round-r receivers (cumulative co-hop interference). Returns the
    per-pair bottleneck (min-over-hops) rate list."""
    m = len(paths)
    K = max(len(p) - 1 for p in paths)
    min_rate = [float("inf")] * m
    for r in range(1, K + 1):
        active = []
        for i in range(m):
            if len(paths[i]) - 1 >= r:
                active.append((i, paths[i][r - 1], paths[i][r]))
        for (i, tx, rx) in active:
            d = dist(tx, rx)
            signal = P / ((1.0 + d) ** alpha)
            interf = 0.0
            for (j, txj, rxj) in active:
                if j == i:
                    continue
                dj = dist(txj, rx)
                interf += P / ((1.0 + dj) ** alpha)
            sinr = signal / (N0 + interf)
            rate = math.log2(1.0 + sinr)
            if rate < min_rate[i]:
                min_rate[i] = rate
    return min_rate


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        p = 0
        m = int(itoks[p]); p += 1
        R = int(itoks[p]); p += 1
        P = float(itoks[p]); p += 1
        alpha = float(itoks[p]); p += 1
        N0 = float(itoks[p]); p += 1
        Xmax = float(itoks[p]); p += 1
        Ymax = float(itoks[p]); p += 1
        SD = []
        for _ in range(m):
            sx = float(itoks[p]); p += 1
            sy = float(itoks[p]); p += 1
            dx = float(itoks[p]); p += 1
            dy = float(itoks[p]); p += 1
            SD.append(((sx, sy), (dx, dy)))
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if not otoks:
        fail("empty output")

    # ---- parse participant artifact: m lines "k x1 y1 ... xk yk" ----
    q = 0
    total_relays = 0
    paths = []
    for i in range(m):
        if q >= len(otoks):
            fail("truncated output (pair %d)" % i)
        try:
            k = int(otoks[q]); q += 1
        except Exception:
            fail("bad relay count (pair %d)" % i)
        if k < 0:
            fail("negative relay count (pair %d)" % i)
        need = 2 * k
        if q + need > len(otoks):
            fail("truncated relay coords (pair %d)" % i)
        relays = []
        for j in range(k):
            try:
                x = float(otoks[q]); y = float(otoks[q + 1])
            except Exception:
                fail("bad relay coord (pair %d, relay %d)" % (i, j))
            q += 2
            if not (math.isfinite(x) and math.isfinite(y)):
                fail("non-finite relay coord (pair %d, relay %d)" % (i, j))
            if x < -TOL or x > Xmax + TOL or y < -TOL or y > Ymax + TOL:
                fail("relay outside deployment area (pair %d, relay %d)" % (i, j))
            relays.append((x, y))
        total_relays += k
        S, D = SD[i]
        paths.append(build_path(S, D, relays))

    if total_relays > R:
        fail("total relay budget exceeded (%d > %d)" % (total_relays, R))

    # trailing garbage tokens are tolerated (ignored), matching common
    # stdout-artifact conventions elsewhere in this corpus.

    rates = bottleneck_rates(paths, P, alpha, N0)
    for r in rates:
        if not math.isfinite(r):
            fail("non-finite rate")
    F = min(rates)

    # ---- internal trivial baseline: EVERY pair gets exactly one relay,
    # placed at its own straight-line midpoint (ignores interference/
    # concavity trade-offs entirely; assumes R >= m, guaranteed by gen.py). ----
    base_paths = []
    for (S, D) in SD:
        mid = ((S[0] + D[0]) / 2.0, (S[1] + D[1]) / 2.0)
        base_paths.append(build_path(S, D, [mid]))
    base_rates = bottleneck_rates(base_paths, P, alpha, N0)
    B = min(base_rates)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
