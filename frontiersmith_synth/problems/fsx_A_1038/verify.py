import sys, math, cmath


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]


def field(N, P, chosen, j):
    s = 0j
    for (i, p) in chosen:
        s += cmath.exp(2j * math.pi * (p / P + i * j / N))
    return s


def intensity(N, P, chosen, j):
    v = field(N, P, chosen, j)
    return v.real * v.real + v.imag * v.imag


def main():
    inp = open(sys.argv[1]).read().split()
    outraw = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        N = int(next(it)); K = int(next(it)); P = int(next(it))
        T = int(next(it))
        targets = [int(next(it)) for _ in range(T)]
        Q = int(next(it))
        protected = []
        thresholds = {}
        for _ in range(Q):
            q = int(next(it)); th = float(next(it))
            protected.append(q); thresholds[q] = th
    except Exception:
        fail("bad input")

    if N <= 0 or K <= 0 or P <= 0 or T <= 0:
        fail("bad input params")

    # ---- parse participant output: m, then m pairs (site phase) ----
    try:
        it2 = iter(outraw)
        m = int(next(it2))
        if m < 1 or m > K:
            fail("count out of [1,K]")
        chosen = []
        for _ in range(m):
            i = int(next(it2))
            p = int(next(it2))
            chosen.append((i, p))
    except Exception:
        fail("parse")

    seen = set()
    for (i, p) in chosen:
        if not (0 <= i < N):
            fail("site out of range %r" % i)
        if not (0 <= p < P):
            fail("phase out of range %r" % p)
        if i in seen:
            fail("duplicate site %r" % i)
        seen.add(i)

    # ---- hard feasibility: every protected bearing must stay dark ----
    for q in protected:
        I = intensity(N, P, chosen, q)
        if not math.isfinite(I):
            fail("non-finite field")
        if I > thresholds[q] + 1e-6:
            fail("protected bearing %d lit: I=%.6f > thr=%.6f" % (q, I, thresholds[q]))

    # ---- objective: worst-illuminated harbor ----
    F = min(intensity(N, P, chosen, j) for j in targets)

    # ---- internal baseline B: best single "comb" null-placement construction the
    # checker can find on its own -- a divisor d of N (d<=K) whose comb (d equally
    # spaced same-phase emitters) divides every harbor bin and no protected bin. ----
    B = 0.0
    for dd in divisors(N):
        if dd > K:
            continue
        if any(t % dd != 0 for t in targets):
            continue
        if any(q % dd == 0 for q in protected):
            continue
        val = float(dd * dd)
        if val > B:
            B = val
    if B <= 0:
        fail("checker baseline unavailable (should not happen)")

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
