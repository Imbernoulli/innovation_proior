import sys
import math

MAX_CHAINS = 6000
MAX_SUM_L = 3_000_000


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad checker invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        in_tokens = f.read().split()
    it = iter(in_tokens)
    p = int(next(it))
    alpha = int(next(it))
    m = int(next(it))
    T = [int(next(it)) for _ in range(m)]
    Tset = set(T)

    try:
        with open(out_path) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")

    toks = raw.split()
    pos = 0

    def next_int():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("truncated output")
        tok = toks[pos]
        pos += 1
        # int() raises ValueError on 'nan'/'inf'/floats/garbage -> caught below (rejects
        # non-finite / non-integer participant output before any scoring happens)
        return int(tok)

    try:
        c = next_int()
    except Exception:
        fail("could not parse chain count")

    if c < 1 or c > MAX_CHAINS:
        fail("chain count %d out of bounds [1, %d]" % (c, MAX_CHAINS))

    chains = []
    sum_L = 0
    try:
        for _ in range(c):
            a = next_int()
            r = next_int()
            L = next_int()
            if not (1 <= a <= p - 1):
                fail("anchor a=%d out of range [1,%d]" % (a, p - 1))
            if not (1 <= r <= p - 1):
                fail("ratio r=%d out of range [1,%d]" % (r, p - 1))
            if not (1 <= L <= p - 1):
                fail("length L=%d out of range [1,%d]" % (L, p - 1))
            sum_L += L
            if sum_L > MAX_SUM_L:
                fail("total sweep length exceeds %d" % MAX_SUM_L)
            chains.append((a, r, L))
    except ValueError:
        fail("non-integer / non-finite token in output")
    except IndexError:
        fail("truncated output")

    if pos != len(toks):
        fail("trailing garbage after the declared %d chains" % c)

    covered = set()
    for a, r, L in chains:
        val = a
        for _ in range(L):
            covered.add(val)
            val = (val * r) % p

    missing = Tset - covered
    if missing:
        fail("target not fully covered, %d villages still dark" % len(missing))

    F = sum_L + alpha * c
    if F <= 0:
        fail("non-positive cost")

    B = m * (1 + alpha)  # checker's own trivial construction: one singleton chain per village
    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    if not math.isfinite(ratio):
        fail("non-finite ratio computed")
    print("F=%d B=%d chains=%d" % (F, B, c))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
