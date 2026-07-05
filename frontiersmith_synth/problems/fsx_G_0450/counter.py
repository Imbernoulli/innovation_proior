import sys

# Format D checker -- reversible (MCT) circuit synthesis for a fixed permutation.
#   1) Parse target permutation pi over n bits from <in>.
#   2) Parse participant's MCT gate list from <out> (strict schema, non-finite rejected).
#   3) EXACT-equivalence gate: simulate the circuit on all 2^n inputs; every output
#      must equal pi(x), else Ratio 0.0.
#   4) Objective (minimize) = gate count F. Baseline B = gate count of the checker's
#      own basic transformation-based (MMD) synthesis of the SAME permutation.
#      Ratio = min(1, 0.1 * B / F).
#
# Deterministic, integer-only, no timing.

GMAX = 50000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


# ---- checker's own baseline: basic transformation-based (MMD) synthesis ----
# Reduces t = pi^{-1} to the identity using output-side (left-compose) MCT gates,
# fixing rows 0,1,2,... in order. The gate list, applied in append order, computes pi.
def synth_basic(perm, n):
    N = 1 << n
    t = [0] * N
    for x in range(N):
        t[perm[x]] = x            # t = pi^{-1}
    count = 0
    for i in range(N):
        # step A: set bits present in i but absent in t[i]
        ns = i & ~t[i]
        b = 0
        while ns:
            if ns & 1:
                cmask = t[i]      # controls = set bits of current value
                mask = 1 << b
                for x in range(N):
                    if (t[x] & cmask) == cmask:
                        t[x] ^= mask
                count += 1
            ns >>= 1
            b += 1
        # step B: clear bits present in t[i] but absent in i
        nc = t[i] & ~i
        b = 0
        while nc:
            if nc & 1:
                cmask = i         # controls = set bits of i
                mask = 1 << b
                for x in range(N):
                    if (t[x] & cmask) == cmask:
                        t[x] ^= mask
                count += 1
            nc >>= 1
            b += 1
    return count


def main():
    inp = open(sys.argv[1]).read().split()
    it = iter(inp)
    try:
        n = int(next(it))
    except Exception:
        fail("bad n")
    if not (1 <= n <= 8):
        fail("n out of range")
    N = 1 << n
    perm = [0] * N
    try:
        for x in range(N):
            perm[x] = int(next(it))
    except Exception:
        fail("bad permutation table")
    if sorted(perm) != list(range(N)):
        fail("input is not a permutation")

    B = synth_basic(perm, n)
    if B <= 0:
        fail("degenerate target (identity)")

    # ---- parse participant output (strict, flat token stream) ----
    tok = open(sys.argv[2]).read().split()
    if not tok:
        fail("empty output")
    idx = 0
    try:
        G = int(tok[0])
    except Exception:
        fail("bad G")
    idx = 1
    if G < 0:
        fail("G < 0")
    if G > GMAX:
        fail("G too large")

    gates = []
    for _ in range(G):
        if idx >= len(tok):
            fail("truncated: missing gate control-count")
        try:
            k = int(tok[idx])
        except Exception:
            fail("bad control-count / non-finite token")
        idx += 1
        if not (0 <= k <= n - 1):
            fail("bad control-count value")
        if idx >= len(tok):
            fail("truncated: missing target")
        try:
            tgt = int(tok[idx])
        except Exception:
            fail("bad target / non-finite token")
        idx += 1
        if not (0 <= tgt < n):
            fail("target out of range")
        controls = []
        for _c in range(k):
            if idx >= len(tok):
                fail("truncated: missing control")
            try:
                c = int(tok[idx])
            except Exception:
                fail("bad control / non-finite token")
            idx += 1
            if not (0 <= c < n):
                fail("control out of range")
            if c == tgt:
                fail("control equals target")
            controls.append(c)
        if len(set(controls)) != len(controls):
            fail("duplicate control")
        cmask = 0
        for c in controls:
            cmask |= (1 << c)
        gates.append((tgt, cmask))
    if idx != len(tok):
        fail("trailing tokens after %d gates" % G)

    # ---- exact-equivalence gate: simulate all inputs ----
    for x in range(N):
        y = x
        for (tgt, cmask) in gates:
            if (y & cmask) == cmask:
                y ^= (1 << tgt)
        if y != perm[x]:
            fail("circuit computes wrong permutation (input %d)" % x)

    F = G
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
