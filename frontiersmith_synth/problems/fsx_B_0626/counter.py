import sys

# counter.py -- Format D checker for "conjugated-permutation reversible synthesis".
#
#   <in>:  n
#          pi[0] pi[1] ... pi[N-1]                       (N = 2^n, the target permutation)
#   <out>: L
#          L gate lines, each "k c1 p1 c2 p2 ... ck pk t" -- a generalized (mixed-polarity)
#          controlled-NOT: flip register bit t iff, for every (c_i,p_i), register bit c_i
#          equals p_i.  k=0 is a NOT, k=1 a CNOT, k=2 a Toffoli, k>=3 a wider gate.
#          weight(k) = 1 (k=0), 2 (k=1), 5 (k>=2).
#
# 1) Parse & validate the gate list strictly (bounds, distinctness, finiteness) -> else Ratio 0.
# 2) Simulate the WHOLE circuit on all N register values (numpy-vectorized bit ops); must
#    reproduce pi EXACTLY on every input -> else Ratio 0.
# 3) Objective (minimize) = total weighted gate count F.
#    Baseline B = Q * (weighted cost of a canonical, structure-blind "star + fixed full bit
#    order Gray path" construction), Q=0.6 -- computed here directly from pi, independent of
#    the submission.  Ratio = min(1, 0.1*B/F).

import numpy as np

MAXGATES = 20000
Q = 0.6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def gate_weight(k):
    if k == 0:
        return 1
    if k == 1:
        return 2
    return 5


# ---------------- baseline: canonical structure-blind construction ----------------
def canonical_baseline_cost(pi, n):
    N = 1 << n
    cur = list(range(N))
    pos = list(range(N))
    total = 0
    per_gate_w = gate_weight(n - 1) if n - 1 >= 0 else 1
    fixed_len = 2 * n - 1 if n >= 1 else 0
    for x in range(N):
        target = pi[x]
        if cur[x] == target:
            continue
        y = pos[target]
        total += fixed_len * per_gate_w
        vx, vy = cur[x], cur[y]
        cur[x], cur[y] = cur[y], cur[x]
        pos[vx], pos[vy] = y, x
    return total


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        n = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= n <= 20):
        fail("bad n")
    N = 1 << n
    try:
        pi = [int(next(it)) for _ in range(N)]
    except Exception:
        fail("bad permutation table")
    if sorted(pi) != list(range(N)):
        fail("degenerate instance (not a permutation) -- generator bug")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    oit = iter(out)
    try:
        L = int(next(oit))
    except Exception:
        fail("bad L")
    if L < 0:
        fail("L < 0")
    if L > MAXGATES:
        fail("L too large (> %d)" % MAXGATES)

    gates = []  # (controls: list[(bit,pol)], target)
    try:
        for _ in range(L):
            k = int(next(oit))
            if not (0 <= k <= n - 1):
                fail("gate control count out of range")
            controls = []
            seen = set()
            for _ in range(k):
                c = int(next(oit))
                p = int(next(oit))
                if not (0 <= c < n):
                    fail("control bit out of range")
                if p not in (0, 1):
                    fail("control polarity not 0/1")
                if c in seen:
                    fail("duplicate control bit")
                seen.add(c)
                controls.append((c, p))
            t = int(next(oit))
            if not (0 <= t < n):
                fail("target bit out of range")
            if t in seen:
                fail("target coincides with a control")
            gates.append((controls, t))
    except SystemExit:
        raise
    except Exception:
        fail("malformed gate token (non-integer / truncated / nan / inf)")
    # reject any leftover trailing tokens beyond the declared gate count? (be lenient: ignore)

    # ---- simulate on all N basis states (vectorized) ----
    reg = np.arange(N, dtype=np.int64)
    for controls, t in gates:
        mask = np.ones(N, dtype=bool)
        for c, p in controls:
            bit = (reg >> c) & 1
            mask &= (bit == p)
        reg = np.where(mask, reg ^ (1 << t), reg)

    target_arr = np.array(pi, dtype=np.int64)
    if not np.array_equal(reg, target_arr):
        bad = int(np.argmax(reg != target_arr))
        fail("circuit does not reproduce the target permutation (first mismatch at x=%d)" % bad)

    F = sum(gate_weight(len(controls)) for controls, _ in gates)
    if F <= 0:
        fail("zero-cost circuit cannot be correct for a non-identity permutation")

    Braw = canonical_baseline_cost(pi, n)
    B = Q * Braw

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("n=%d L=%d F=%d B=%.3f Ratio: %.6f" % (n, L, F, B, ratio))


if __name__ == "__main__":
    main()
