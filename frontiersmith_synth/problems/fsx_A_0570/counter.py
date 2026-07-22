import sys

# Format D checker -- "one multiplier ladder for many exponents".
#
# The participant submits a single ADDITION CHAIN (a straight-line program of
# doublings/additions) that must PRODUCE every target integer, and is scored on
# the number of addition steps L (fewer = better).
#
#   1) Parse the batch of k targets from <in>.
#   2) Parse the participant's chain from <out>:
#          L
#          a_1 b_1
#          ...
#          a_L b_L
#      value[0] = 1.  Step i (1-based) sets value[i] = value[a_i] + value[b_i]
#      with 0 <= a_i, b_i < i.  The set of produced values must contain every
#      target (EXACT-equivalence gate; any miss -> Ratio 0.0).
#   3) Objective (minimize) = L.  Baseline B = the cost of the INDEPENDENT
#      square-and-multiply method (each target with its own binary chain):
#          B = sum_j (bitlen(n_j)-1) + (popcount(n_j)-1).
#      Ratio = min(1, 0.1 * B / L).   trivial ~ 0.1 ; 10x fewer ops caps at 1.0.

MAXL = 100000          # hard cap on chain length (checker stays O(L))
MAXBITS = 4096         # abuse guard: reject absurdly large intermediate values


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("no input")
    it = iter(inp)
    try:
        k = int(next(it))
        targets = [int(next(it)) for _ in range(k)]
    except Exception:
        fail("bad instance")
    if k < 1 or any(t < 1 for t in targets):
        fail("bad targets")

    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not out:
        fail("empty output")

    jt = iter(out)
    try:
        L = int(next(jt))
    except Exception:
        fail("bad L")
    if L < 0 or L > MAXL:
        fail("L out of range")

    values = [1]
    produced = {1}
    try:
        for i in range(1, L + 1):
            a = int(next(jt))
            b = int(next(jt))
            # value list currently holds indices 0..i-1
            if a < 0 or b < 0 or a >= i or b >= i:
                fail("index out of range at step %d" % i)
            v = values[a] + values[b]
            if v.bit_length() > MAXBITS:
                fail("intermediate value too large")
            values.append(v)
            produced.add(v)
    except StopIteration:
        fail("chain truncated (need %d steps)" % L)
    except Exception:
        fail("bad step token")

    # EXACT equivalence: every target must be a produced value.
    for t in targets:
        if t not in produced:
            fail("target %d not produced" % t)

    F = L if L > 0 else 1
    B = 0
    for t in targets:
        B += (t.bit_length() - 1) + (bin(t).count("1") - 1)
    if B < 1:
        B = 1

    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("L=%d B=%d Ratio: %.6f" % (L, B, ratio))


if __name__ == "__main__":
    main()
