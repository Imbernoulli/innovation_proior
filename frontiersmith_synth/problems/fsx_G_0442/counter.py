#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  -- deterministic scorer for the symmetric-function
logic-synthesis problem (Format D, eval_form=flops).

Pipeline:
  1. Read the instance (n, accepting popcount set A).
  2. Parse the participant's straight-line AND/OR/NOT circuit under a STRICT schema
     (bounded, integer-only, backward references only). Any violation -> Ratio: 0.0.
  3. Verify EXACT equivalence to f on ALL 2^n inputs via bit-parallel simulation
     (each wire = a 2^n-bit integer). Mismatch -> Ratio: 0.0.
  4. Count gates F. Internal baseline B = a full bubble sorting network + naive
     exactly-k decode (a legitimate feasible construction). Minimization:
        Ratio = min(1.0, 0.1 * B / max(1, F))
  Everything is exact integer arithmetic; nothing is timed.
"""
import sys

GATE_CAP = 2000       # bounds simulation memory/time; legit circuits use < 500 gates


def read_input(path):
    toks = open(path).read().split()
    n = int(toks[0])
    spec = toks[1]
    A = [k for k in range(n + 1) if k < len(spec) and spec[k] == '1']
    return n, A


def build_columns(n):
    """Input-bit truth-table columns as 2^n-bit integers (bit j set iff input j has bit i)."""
    N = 1 << n
    mask = (1 << N) - 1
    cols = []
    for i in range(n):
        block = 1 << i
        val = ((1 << block) - 1) << block   # one period: 0-run then 1-run, width 2*block
        width = 2 * block
        while width < N:
            val |= val << width             # replicate by doubling
            width <<= 1
        cols.append(val & mask)
    return cols, mask, N


def build_target(n, A, cols, mask):
    """Ground-truth output vector: bit j set iff popcount(j) in A. Computed by
    simulating a sorting network on the column bit-vectors (thresholds), so it is
    exactly the symmetric function f."""
    wire = cols[:]
    for a in range(n - 1):
        for j in range(n - 1 - a):
            x, y = wire[j], wire[j + 1]
            wire[j] = x & y      # min
            wire[j + 1] = x | y  # max
    # ascending sort: wire[p] == 1  iff  popcount >= n-p  => threshold_k = wire[n-k]

    def thr(k):
        if k <= 0:
            return mask
        if k > n:
            return 0
        return wire[n - k]

    T = 0
    for k in A:
        T |= thr(k) & (~thr(k + 1) & mask)
    return T


def parse_circuit(path, n):
    toks = open(path).read().split()
    it = iter(toks)
    try:
        G = int(next(it))
    except StopIteration:
        return None, "empty"
    except ValueError:
        return None, "non-integer gate count"
    if G < 0 or G > GATE_CAP:
        return None, "gate count out of range"
    gates = []
    for gi in range(G):
        cur = n + gi
        try:
            op = next(it)
        except StopIteration:
            return None, "truncated gate list"
        if op in ("AND", "OR"):
            try:
                a = int(next(it)); b = int(next(it))
            except StopIteration:
                return None, "missing operand"
            except ValueError:
                return None, "non-integer operand"
            if not (0 <= a < cur and 0 <= b < cur):
                return None, "operand references non-earlier wire"
            gates.append((op, a, b))
        elif op == "NOT":
            try:
                a = int(next(it))
            except StopIteration:
                return None, "missing operand"
            except ValueError:
                return None, "non-integer operand"
            if not (0 <= a < cur):
                return None, "operand references non-earlier wire"
            gates.append(("NOT", a, a))
        else:
            return None, "unknown op '%s'" % op[:16]
    try:
        kw = next(it)
    except StopIteration:
        return None, "missing OUT line"
    if kw != "OUT":
        return None, "expected OUT"
    try:
        w = int(next(it))
    except StopIteration:
        return None, "missing OUT wire"
    except ValueError:
        return None, "non-integer OUT wire"
    if not (0 <= w < n + G):
        return None, "OUT wire out of range"
    if list(it):
        return None, "trailing tokens"
    return (gates, w), None


def evaluate(gates, out, cols, mask):
    val = cols[:]
    for op, a, b in gates:
        if op == "AND":
            val.append(val[a] & val[b])
        elif op == "OR":
            val.append(val[a] | val[b])
        else:
            val.append((~val[a]) & mask)
    return val[out]


def baseline_count(n, A):
    def cost(k):
        if k == n:
            return 0
        if k == 0:
            return 1
        return 2
    return n * (n - 1) + sum(cost(k) for k in A) + max(0, len(A) - 1)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    n, A = read_input(inp)
    cols, mask, N = build_columns(n)
    T = build_target(n, A, cols, mask)

    res, err = parse_circuit(outp, n)
    if err is not None:
        print("infeasible circuit (%s) Ratio: 0.0" % err)
        return
    gates, out = res
    got = evaluate(gates, out, cols, mask)
    if got != T:
        print("circuit does not compute f (truth-table mismatch) Ratio: 0.0")
        return

    F = len(gates)
    B = baseline_count(n, A)
    sc = min(1.0, 0.1 * B / max(1, F))
    print("gates=%d baseline=%d Ratio: %.6f" % (F, B, sc))


if __name__ == "__main__":
    main()
