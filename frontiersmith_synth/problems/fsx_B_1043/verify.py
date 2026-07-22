#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for "Welding a Frame That Flinches at Every Joint".

Reads the frame from <in>, the weld order + side choices from <out>, simulates
the exact stiffness-discounted accumulation, and scores against the checker's
own reference (input-order, all sides +1) baseline. Prints "Ratio: <float>"
on its own final line and exits 0.
"""
import math
import sys


def fail(msg):
    sys.stderr.write(msg + "\n")
    print("Ratio: 0.0")
    sys.exit(0)


EFF_BOUND = 100  # matches statement.md's "1 <= |eff_i| <= 100"


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        n = int(next(it))
        m = int(next(it))
        edges = []
        for _ in range(m):
            u = int(next(it))
            w = int(next(it))
            eff = int(next(it))
            if eff == 0 or abs(eff) > EFF_BOUND:
                raise RuntimeError(f"eff={eff} outside 1<=|eff|<={EFF_BOUND}")
            edges.append((u, w, eff))
    except (StopIteration, ValueError) as e:
        raise RuntimeError(f"malformed instance: {e}")
    return n, m, edges


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.weld_count = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            self.weld_count[ra] += 1
            return
        # union by attaching smaller root under larger, merged weld count
        merged = self.weld_count[ra] + self.weld_count[rb] + 1
        self.p[rb] = ra
        self.weld_count[ra] = merged


def simulate(n, edges, order, sides):
    """order: list of edge indices (weld sequence). sides: aligned list of +-1.
    Returns list of final displacements per node."""
    dsu = DSU(n)
    disp = [0.0] * n
    for e_idx, s in zip(order, sides):
        u, w, eff = edges[e_idx]
        ru, rw = dsu.find(u), dsu.find(w)
        stiff_u = dsu.weld_count[ru]
        stiff_w = dsu.weld_count[rw]
        disp[u] += s * eff / (1.0 + stiff_u)
        disp[w] -= s * eff / (1.0 + stiff_w)
        dsu.union(u, w)
    return disp


def parse_output(path, m):
    with open(path) as f:
        toks = f.read().split()
    if not toks:
        raise RuntimeError("empty output")
    it = iter(toks)
    try:
        m_out = int(next(it))
    except ValueError:
        raise RuntimeError("first token not an integer")
    if m_out != m:
        raise RuntimeError(f"declared M={m_out} != instance M={m}")
    order = []
    sides = []
    for _ in range(m):
        try:
            e_tok = next(it)
            s_tok = next(it)
        except StopIteration:
            raise RuntimeError("fewer than M (edge,side) pairs")
        # strict integer parse -- rejects "1.0", "nan", "inf", etc.
        try:
            e = int(e_tok)
            s = int(s_tok)
        except ValueError:
            raise RuntimeError("non-integer token in weld order/side")
        order.append(e)
        sides.append(s)
    # reject trailing garbage tokens
    rest = list(it)
    if rest:
        raise RuntimeError("trailing tokens after M pairs")
    if len(order) != m or sorted(order) != list(range(m)):
        raise RuntimeError("edge indices are not a permutation of 0..M-1")
    if any(s not in (1, -1) for s in sides):
        raise RuntimeError("side must be exactly 1 or -1")
    return order, sides


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    try:
        n, m, edges = read_instance(inp)
    except Exception as e:
        # a malformed instance is our own bug, not a scoring case; fail loudly
        sys.stderr.write(f"bad instance: {e}\n")
        print("Ratio: 0.0")
        return

    try:
        order, sides = parse_output(outp, m)
    except Exception as e:
        fail(f"invalid submission: {e}")

    # defense in depth: eff is already bounds-checked in read_instance, so
    # this cannot overflow, but never let a submission-triggered exception
    # escape as a nonzero exit (which the harness would also score 0, but
    # a graceful "Ratio: 0.0" is the documented contract).
    try:
        disp = simulate(n, edges, order, sides)
        if not all(math.isfinite(x) for x in disp):
            fail("non-finite displacement")
        F = max(abs(x) for x in disp)

        # reference baseline: weld in input order, every side +1
        base_disp = simulate(n, edges, list(range(m)), [1] * m)
        B = max(abs(x) for x in base_disp)
    except (OverflowError, ArithmeticError, ValueError) as e:
        fail(f"arithmetic error during simulation: {e}")

    if not math.isfinite(B) or B <= 0.0:
        fail("degenerate baseline (should not happen for a valid instance)")

    if not math.isfinite(F):
        fail("non-finite objective")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
