#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (Format D: op-count / addition-chain length)

The participant submits a straight-line program of additions from r_0 = 1:
  line 1: L            number of instructions
  lines 2..L+1: a b    meaning r_i = r_a + r_b  (1-based i, 0 <= a,b < i)

We FIRST verify exact functional equivalence (every target value appears among
the registers, all operand indices valid, every register value <= max target),
THEN count instructions F = L (fewer = better).

Score (minimization): sc = min(1000, 100 * B / F), ratio = sc/1000, where B is
the checker's own baseline = total op count of computing each target
independently with the binary method, cost(e) = floor(log2 e) + popcount(e) - 1.
Any feasibility violation -> Ratio: 0.0. Fully deterministic; no randomness.
"""
import sys

MAX_L = 20000


def fail(reason):
    print("VIOLATION: %s  Ratio: 0.0" % reason)
    sys.exit(0)


def bin_cost(e):
    # floor(log2 e) + popcount(e) - 1  (binary method, no sharing)
    return e.bit_length() - 1 + bin(e).count("1") - 1


def main():
    data = open(sys.argv[1]).read().split()
    it = iter(data)

    def ni():
        return int(next(it))

    K = ni()
    targets = [ni() for _ in range(K)]
    T = max(targets)
    tset = set(targets)

    raw = open(sys.argv[2]).read().split()
    if not raw:
        fail("empty output")
    try:
        L = int(raw[0])
    except ValueError:
        fail("first token (instruction count) is not an integer")
    if L < 0 or L > MAX_L:
        fail("instruction count out of range [0,%d]" % MAX_L)
    if len(raw) != 1 + 2 * L:
        fail("token count mismatch: expected %d operand tokens, got %d"
             % (2 * L, len(raw) - 1))

    regs = [1]
    for i in range(1, L + 1):
        try:
            a = int(raw[2 * i - 1])
            b = int(raw[2 * i])
        except ValueError:
            fail("non-integer operand on line %d" % (i + 1))
        if not (0 <= a < i and 0 <= b < i):
            fail("operand index out of range on instruction %d" % i)
        v = regs[a] + regs[b]
        if v > T:
            fail("register value %d exceeds max target %d" % (v, T))
        regs.append(v)

    rset = set(regs)
    for t in tset:
        if t not in rset:
            fail("target %d not produced" % t)

    F = L
    B = sum(bin_cost(t) for t in tset)
    if B <= 0:
        B = 1  # degenerate safeguard (generator guarantees B > 0)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("ops=%d baseline=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
